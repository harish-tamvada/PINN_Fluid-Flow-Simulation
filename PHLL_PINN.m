%% PHLL_case_1p0 PINN Training
clc
clear

%% =============== Import Data ===============
% Data prefiltered in python and saved as filtered_data.mat
% X, Y co-ords as Cx Cy
% Ux Uy p values for learning
% uu uv vv values for PINN calculations
load("filtered_data.mat");

data = all_data.PHLL_case_1p0;
%% =============== Getting data ready for training ===============

% Know Re = Ub * H / nu
% Therefore by setting Re = 5600 (as per the paper), we can estimate Ub and
% nu for normalisation and physics model

scalers.H = 1; %Cy is already in units of H
scalers.Re = 5600;


% Ub is for the mean velocity @ a crest NOT over the whole domain 
% Find which x points are at the hill crest (per the paper x = 0 is a
% crest) and then find the velocity mean for those points only

x = data.Cx;
crestX = min(x);           % x = 0 plane, at the first hill crest
xTol = 0.05;
mask = abs(x - crestX) < xTol;
if any(mask)
    scalers.Ub = mean(data.Ux(mask));
else
    scalers.Ub = mean(abs(data.Ux));
end

%From rearrangement of Re equation
scalers.nu = scalers.Ub * scalers.H / scalers.Re;

fprintf("The predicted kinematic viscosity is : %.2d m^2/s", scalers.nu);

%For scaling to a [-1 1] domain
scalers.xmax = max(data.Cx);
scalers.ymax = max(data.Cy);

%X Y ready for training
X = 2 * (data.Cx / scalers.H) / scalers.xmax  - 1;
Y = 2 * (data.Cy / scalers.H) / scalers.ymax  - 1;

%Ux Uy p uu uv vv ready for training
Ub2 = scalers.Ub^2;
uStar  = data.Ux / scalers.Ub;
vStar  = data.Uy / scalers.Ub;
pStar  = data.p  / Ub2;
uuStar = data.uu / Ub2;
uvStar = data.uv / Ub2;
vvStar = data.vv / Ub2;

%% =============== Building BC ===============

% To get y contour at base, bin Cx and find min Cy value for given Cx and
% build contour 

nBins = 100;

x = data.Cx;
y = data.Cy;

edges = linspace(min(x), max(x), nBins + 1);
xWall = zeros(nBins, 1);
yWall = zeros(nBins, 1);
keep = true(nBins, 1);


% Bin the data to find the minimum y value for each bin
for i = 1:nBins
    binMask = (x >= edges(i)) & (x < edges(i + 1));
    if any(binMask)
        yWall(i) = min(y(binMask));
        xWall(i) = mean(x(binMask));
    else
        keep(i) = false; % Mark bins with no data
    end
end

% Remove empty bins
xWall = xWall(keep);
yWall = yWall(keep);


% Top wall is y = ymax 
xTop = linspace(0, scalers.xmax, nBins)';
yTop = scalers.ymax * ones(nBins, 1);


%combine both wall BCs
xBC = [xWall; xTop];
yBC = [yWall; yTop];

%non-dim wall BC
XBC = 2 * (xBC / scalers.H) / scalers.xmax  - 1;
YBC = 2 * (yBC / scalers.H) / scalers.ymax  - 1;

% NOTE: No explicit IC for this model

%% =============== Fourier Feature Hyperparameters ===============

ff = 128;    % no. of Fourier Features
gamma = 1;   %scalar (keep at 1)
d = 2;       %input dimension (x,y)

% B formation
rng(0);
B = gamma * randn(ff, d);

%% =============== Neural Network Hyperparameters ===============

%Size of network
numBlocks = 5;       % no. of hidden layers
layerSize = 80;     % nodes per hidden layer

% Normal Distribution SF for RWF
sigma = 0.1;
mu = 1.0;

% Input/Output sizing
inputDim = 2 * ff;
outputDim = 3;

%% =============== Neural Network Formation ===============

net = buildNetwork(inputDim, outputDim, layerSize, numBlocks, mu, sigma);

if canUseGPU
    B = gpuArray(single(B));
end
%% =============== Physical collocation points + Reynolds Stress terms ===============

% Interpolate the Reynolds Stresses
uuInterp = scatteredInterpolant(X', Y', uuStar', 'natural', 'nearest');
uvInterp = scatteredInterpolant(X', Y', uvStar', 'natural', 'nearest');
vvInterp = scatteredInterpolant(X', Y', vvStar', 'natural', 'nearest');

% Generate x,y mesh grid that is within wall domain

%300 *200 mesh
xPoints = linspace(0, scalers.xmax, 300);
yPoints = linspace(0, scalers.ymax, 200);
[Xmesh, Ymesh] = meshgrid(xPoints, yPoints);
Xmesh = Xmesh(:);
Ymesh = Ymesh(:);

% wall height at each grid x-location, via 1D interpolation of the
% (already extracted) wall contour
[xWallSorted, sortIdx] = sort(XBC);
yWallSorted = YBC(sortIdx);
wallY = interp1(xWallSorted, yWallSorted, Xmesh, 'linear', 'extrap');

validMask = Ymesh > wallY;   % keep only points ABOVE the wall (in the fluid)

xStar = Xmesh(validMask);
yStar = Ymesh(validMask);

% Calculate Reynold stress gradients @ mesh grid points using numerical
% methods
h = 1e-3;
dUUdx = (uuInterp(xStar + h, yStar) - uuInterp(xStar - h, yStar)) / (2*h);
dUVdy = (uvInterp(xStar, yStar + h) - uvInterp(xStar, yStar - h)) / (2*h);
dUVdx = (uvInterp(xStar + h, yStar) - uvInterp(xStar - h, yStar)) / (2*h);
dVVdy = (vvInterp(xStar, yStar + h) - vvInterp(xStar, yStar - h)) / (2*h);

%non-dim x,y mesh grid
XphysNet = 2 * (xStar / scalers.H) / scalers.xmax  - 1;
YphysNet = 2 * (yStar / scalers.H) / scalers.xmax  - 1;


%% =============== Create dlarrays for fixed data ===============

% 3 loss functions: Physics Informed loss, Loss to u,v,p data known,
% Loss due to BC
% Physics Informed Loss takes minibatches of a random mesh grid (300 x 200)
% BC: whole array
% Data: random selection from ~14750 data array


% Create dlarrays for boundary conditions and physics check mesh grid
bcX = convertArray(XBC');
bcY = convertArray(YBC');

physX = convertArray(XphysNet');
physY = convertArray(YphysNet');

%% =============== Adams Training Settings ===============

maxnumIt     = 5000;
batchSizeData = 2000;
learnRate     = 1e-3;
lambdaPhys    = 5.0;     % TUNE: weight on physics residual
lambdaBC      = 0.75;    % TUNE: weight on wall BC
%Want improvements in the physics loss to be 'worth' a lot more than other
%gains

% Set up of the minibatch of the DNS data
numDataPoints = numel(X);
numBatches    = floor(numDataPoints / batchSizeData);
 
averageGrad   = [];
averageSqGrad = [];
iteration     = 0;
 
lossHistory = zeros(maxnumIt, 4);  % [total, data, physics, bc]

for epoch = 1:maxnumIt
    shuffledIdx = randperm(numDataPoints);
 
    epochLoss = zeros(1,4);
 
    for b = 1:numBatches
        iteration = iteration + 1;

        % Selecting minibatch and converting to dlarray
        batchIdx = shuffledIdx((b-1)*batchSizeData + 1 : b*batchSizeData);
 
        XdataBatch = convertArray(X(batchIdx));
        YdataBatch = convertArray(Y(batchIdx));
        uBatch     = convertArray(uStar(batchIdx));
        vBatch     = convertArray(vStar(batchIdx));
        pBatch     = convertArray(pStar(batchIdx));

        %Loss funtion
      [loss, gradients, lD, lP, lB] = dlfeval(@modelLoss, net, ...
            XdataBatch, YdataBatch, uBatch, vBatch, pBatch, ...
            physX, physY, dUUdx, dUVdy, dUVdx, dVVdy, scalers.Re, scalers, ...
            bcX, bcY, lambdaPhys, lambdaBC, B);
        
        % Grad descent updates
        [net, averageGrad, averageSqGrad] = adamupdate(net, gradients, ...
            averageGrad, averageSqGrad, iteration, learnRate);

        % Loss calculation (cumulative loss for each batch)
        epochLoss = epochLoss + [extractdata(loss), extractdata(lD), extractdata(lP), extractdata(lB)];
    end
    
    % Avge loss per iteration (we do numBatches of simulations per
    % iteration)
    lossHistory(epoch, :) = epochLoss / numBatches;
    
    % Debug output
    if mod(epoch, 50) == 0 || epoch == 1
        fprintf("Epoch %4d | total=%.6f  data=%.6f  physics=%.6f  bc=%.6f\n", ...
            epoch, lossHistory(epoch,1), lossHistory(epoch,2), ...
            lossHistory(epoch,3), lossHistory(epoch,4));
    end
end
 
%% =============== Loss Plot Curve ===============
trainingHistoryFig = figure;

semilogy(lossHistory);
legend('total','data','physics','bc');
xlabel('epoch'); ylabel('loss (log scale)');
title('PINN training loss');

%% =============== Save Figures and Net for Plotting ===============

% Cannot save net as is, need to save its learnables and hyperparameters
% and use buildNetwork function to rebuild it in another script (plotting
% script)
learnables = net.Learnables;

save('trainedPINN.mat', 'learnables', 'data', 'scalers', 'inputDim', 'outputDim', 'layerSize', 'numBlocks', 'mu', 'sigma', 'B');

outputDir = "PINN_Figures";
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end
    
exportgraphics(trainingHistoryFig, fullfile(outputDir, 'trainingHistory.png') , 'Resolution', 200);
savefig(trainingHistoryFig,  fullfile(outputDir, 'trainingHistory.fig'));
