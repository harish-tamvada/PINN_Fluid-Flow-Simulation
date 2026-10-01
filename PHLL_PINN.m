%% PHLL_case_1p0 PINN Training
clc
clear
close all

runTimeStart = tic;
%% =============== Import Data ===============
% Data prefiltered in python and saved as filtered_data.mat
% X, Y co-ords as Cx Cy
% Ux Uy p values for learning
% uu uv vv values for PINN calculations
% load("filtered_data.mat");
% 
% data = all_data.PHLL_case_1p0;


folder = "Raw_Data";
cases = "PHLL_case_1p0";
fields_komegasst = ["Cx", "Cy", "Ux", "Uy", "p"];
fields_labels = ["uu", "uv", "vv"];

all_data = loadData(folder, cases, fields_komegasst, fields_labels);

data = all_data.PHLL_case_1p0;
%% =============== Getting data ready for training ===============

% Removing NaN/Inf data points
% Stack everything that must stay row-aligned
fields = [data.Cx(:), data.Cy(:), data.Ux(:), data.Uy(:), data.p(:), data.uu(:), data.uv(:), data.vv(:)];

bad = any(isnan(fields) | isinf(fields), 2);   % true if ANY field is NaN/Inf in that row
fprintf("Removing %d of %d rows with NaN/Inf\n", nnz(bad), numel(bad));

data.Cx = data.Cx(~bad);  data.Cy = data.Cy(~bad);
data.Ux = data.Ux(~bad);  data.Uy = data.Uy(~bad);
data.p  = data.p(~bad);   data.uu = data.uu(~bad);
data.uv = data.uv(~bad);  data.vv = data.vv(~bad);

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

fprintf("The predicted kinematic viscosity is : %.3e m^2/s \n", scalers.nu);

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
[xWallSorted, sortIdx] = sort(xWall);
yWallSorted = yWall(sortIdx);
wallY = interp1(xWallSorted, yWallSorted, Xmesh, 'linear', 'extrap');

validMask = Ymesh > wallY + 0.3;   % keep only points ABOVE the wall achieved with 0.3 buffer

xStar = Xmesh(validMask);
yStar = Ymesh(validMask);


%non-dim x,y mesh grid
XphysNet = 2 * (xStar / scalers.H) / scalers.xmax  - 1;
YphysNet = 2 * (yStar / scalers.H) / scalers.ymax  - 1;



% Calculate Reynold stress gradients @ mesh grid points using numerical
% methods - interp of Reynolds stresses uses non-dim x,y co-ords so same
% used here

h = 1e-2;
dUUdX = (uuInterp(XphysNet + h, YphysNet) - uuInterp(XphysNet - h, YphysNet)) / (2*h);
dUVdY = (uvInterp(XphysNet, YphysNet + h) - uvInterp(XphysNet, YphysNet - h)) / (2*h);
dUVdX = (uvInterp(XphysNet + h, YphysNet) - uvInterp(XphysNet - h, YphysNet)) / (2*h);
dVVdY = (vvInterp(XphysNet, YphysNet + h) - vvInterp(XphysNet, YphysNet - h)) / (2*h);


%% =============== Create dlarrays for fixed data ===============

% 3 loss functions: Physics Informed loss, Loss to u,v,p data known,
% Loss due to BC
% Physics Informed Loss takes minibatches of a random mesh grid (300 x 200)
% BC: whole array
% Data: minibatch from ~14750 data array


% Create dlarrays for boundary conditions
if canUseGPU
    bcX = dlarray(gpuArray(single(XBC')), "CB");
    bcY = dlarray(gpuArray(single(YBC')), "CB");
else
    bcX = dlarray(XBC', "CB");
    bcY = dlarray(YBC', "CB");
end


%% =============== Adam Minibatching DNS Data ===============

% 70/30 Train/Validation Split
rng(42);                                   % reproducible split; change or remove as needed

N          = numel(X);
idxShuffle = randperm(N);                  % random permutation of all observations
nTrain     = floor(0.7 * N);

trainIdx = idxShuffle(1:nTrain);
valIdx   = idxShuffle(nTrain+1:end);

% Equivalent logical-mask form
trainMask = false(N,1);
trainMask(trainIdx) = true;
valMask   = ~trainMask;

% Training set
XTrain     = X(trainMask);
YTrain     = Y(trainMask);
uStarTrain = uStar(trainMask);
vStarTrain = vStar(trainMask);
pStarTrain = pStar(trainMask);

% Validation set - creating struct to save as for validation in
% plotModelVsData
valData.XVal       = X(valMask);
valData.YVal       = Y(valMask);
valData.uStarVal   = uStar(valMask);
valData.vStarVal   = vStar(valMask);
valData.pStarVal   = pStar(valMask);

fprintf('RANS split: %d train / %d validation (of %d total)\n', ...
    nnz(trainMask), nnz(valMask), N);

batchSizeDataDNS = 3072;

adsX = arrayDatastore(XTrain(:),     'ReadSize', batchSizeDataDNS);
adsY = arrayDatastore(YTrain(:),     'ReadSize', batchSizeDataDNS);
adsU = arrayDatastore(uStarTrain(:), 'ReadSize', batchSizeDataDNS);
adsV = arrayDatastore(vStarTrain(:), 'ReadSize', batchSizeDataDNS);
adsP = arrayDatastore(pStarTrain(:), 'ReadSize', batchSizeDataDNS);
adsData = combine(adsX, adsY, adsU, adsV, adsP);

mbqData = minibatchqueue(adsData, 5, ...
    "MiniBatchSize",     batchSizeDataDNS, ...
    "MiniBatchFcn",      @rowRescaleFunc, ...
    "MiniBatchFormat",   {'CB','CB','CB','CB','CB'}, ...
    "OutputEnvironment", "auto");

%% =============== Adam Minibatch Physics Data ===============

% want 1 batch of phys data drawn for 1 batch of DNS data i.e. num of
% batches is same

% num of DNS batches
numDataPoints = numel(X);
numBatches    = floor(numDataPoints / batchSizeDataDNS);
% to calculate phys batchh size
numPhysPoints = numel(XphysNet);
batchSizeDataPhys = floor(numPhysPoints/numBatches);



adsPhysX = arrayDatastore(XphysNet(:), 'ReadSize', batchSizeDataPhys);
adsPhysY = arrayDatastore(YphysNet(:), 'ReadSize', batchSizeDataPhys);
adsdUUdx = arrayDatastore(dUUdX(:),    'ReadSize', batchSizeDataPhys);
adsdUVdy = arrayDatastore(dUVdY(:),    'ReadSize', batchSizeDataPhys);
adsdUVdx = arrayDatastore(dUVdX(:),    'ReadSize', batchSizeDataPhys);
adsdVVdy = arrayDatastore(dVVdY(:),    'ReadSize', batchSizeDataPhys);
adsPhys  = combine(adsPhysX, adsPhysY, adsdUUdx, adsdUVdy, adsdUVdx, adsdVVdy);



mbqPhys = minibatchqueue(adsPhys, 6, ...
    "MiniBatchSize",      batchSizeDataPhys, ...
    "MiniBatchFcn", @rowRescaleFunc, ...
    "MiniBatchFormat",    {'CB','CB','CB','CB','CB','CB'}, ...
    "OutputEnvironment",  "auto", ...
    "PartialMiniBatch", "discard"); % ensures batch size is same every iteration


%% =============== Adam Training Initialisation ===============

maxnumItAdam = 500;
lambdaPhys    = 5;     % TUNE weights
lambdaBC      = 0.75;    
lambdaData = 3;
%Want improvements in the physics loss to be 'worth' a lot more than other
%gains

% Learn rate decay
decayRate = 0.95;
learnRateInit = 1e-3;
decaySteps = 250;

 
averageGrad   = [];
averageSqGrad = [];
iteration     = 0;
 
lossHistoryAdam = zeros(maxnumItAdam, 4);  % [total, data, physics, bc]
residualHistoryAdam = zeros(maxnumItAdam, 7); %  [contRes, xMomRes, yMomRes, lossU, lossV, lossPres, relRes]

% Initialised - to store minimum loss achieved such that only the best
% learnables are saved
minLoss = 1e5;

% Accelerated LossFnc    
accFun = dlaccelerate(@modelLoss);
clearCache(accFun);

% moniter initialise
monitor = trainingProgressMonitor( ...
    Metrics = ["TotalLoss", "DataLoss", "PhysicsLoss", "BCLoss"],...
    Info = ["Epoch", "Loss"], ...
    XLabel = "Iteration");
groupSubPlot(monitor, "Losses", ["TotalLoss", "DataLoss", "PhysicsLoss", "BCLoss"]);
yscale(monitor, "Losses", "log");

%% =============== Adam Training Loop ===============
adam_tStart = tic;
for epoch = 1:maxnumItAdam
 
    shuffle(mbqData);
    shuffle(mbqPhys);
    
    % avoid extractdata bottleneck every it and now its every epoch instead
    epochLoss = dlarray(zeros(1,4));
    epochResiduals = dlarray(zeros(1, 7));
 
    batchesPerEpoch = 0;
    while hasdata(mbqData) && hasdata(mbqPhys)
        batchesPerEpoch = batchesPerEpoch + 1;
        iteration = iteration + 1;
 
        % Stepped decay to learn rate
        learnRate = learnRateInit * decayRate^floor(iteration/decaySteps);
 
        % Pull this iteration's minibatches
        [XdataBatch, YdataBatch, uBatch, vBatch, pBatch] = next(mbqData);
        [physXBatch, physYBatch, dUUdxBatch, dUVdyBatch, dUVdxBatch, dVVdyBatch] = next(mbqPhys);
 
        lambdaBCDL   = dlarray(lambdaBC);
        lambdaPhysDL = dlarray(lambdaPhys);
        lambdaDataDL = dlarray(lambdaData);
 
        % Loss function
        [loss, gradients, lossData, lossPhys, lossBC, contRes, xMomRes, yMomRes, lossU, lossV, lossPres, relRes] = dlfeval(accFun, net, ...
            XdataBatch, YdataBatch, uBatch, vBatch, pBatch, ...
            physXBatch, physYBatch, dUUdxBatch, dUVdyBatch, dUVdxBatch, dVVdyBatch, scalers.Re, scalers, ...
            bcX, bcY, lambdaPhysDL, lambdaBCDL, lambdaDataDL, B);
 
        % Grad descent updates
        [net, averageGrad, averageSqGrad] = adamupdate(net, gradients, ...
            averageGrad, averageSqGrad, iteration, learnRate);
 
        % Accumulate cumulative loss for this batch, still as dlarray
        epochLoss = epochLoss + [loss, lossData, lossPhys, lossBC];

        epochResiduals = epochResiduals + [contRes, xMomRes, yMomRes, lossU, lossV, lossPres, relRes];
    end
    numBatchesThisEpoch = batchesPerEpoch;
 
    % Single extractdata call per epoch
    lossHistoryAdam(epoch, :) = extractdata(epochLoss) / numBatchesThisEpoch;
    residualHistoryAdam(epoch, :) = extractdata(epochResiduals) / numBatchesThisEpoch;

    % Debug output
    if mod(epoch, 50) == 0 || epoch == 1
        fprintf("Epoch %4d | total=%.3e  data=%.3e  physics=%.3e  bc=%.6f\n", ...
            epoch, lossHistoryAdam(epoch,1), lossHistoryAdam(epoch,2), ...
            lossHistoryAdam(epoch,3), lossHistoryAdam(epoch,4));
        %--- acceleration cache diagnostics ---
        fprintf("  dlaccelerate hit rate = %.4f  occupancy = %d,\n", ...
            accFun.HitRate, accFun.Occupancy);
        %--- raw loss term diagnostics ---
        fprintf("Physics | ContRes = %.3e xMomRes = %.3e yMomRes=%.3e \nData | uRes=%.3e  vRes=%.3e pRes=%.3e\nRelative Res = %.2f %% \n\n" ,  ...
            residualHistoryAdam(epoch, 1), residualHistoryAdam(epoch, 2), ...
            residualHistoryAdam(epoch, 3), residualHistoryAdam(epoch, 4), ...
            residualHistoryAdam(epoch, 5), residualHistoryAdam(epoch, 6), ...
            residualHistoryAdam(epoch, 7));
    end
 
    % Lowest loss comparison
    if lossHistoryAdam(epoch, 1) < minLoss
        learnables_best = net.Learnables;
        minLoss = lossHistoryAdam(epoch, 1);
    end
 
    % Updating monitor to show model progress
    updateInfo(monitor, ...
        Epoch = epoch, ...
        Loss = lossHistoryAdam(epoch, 1));

    recordMetrics(monitor, epoch, ...
        TotalLoss = lossHistoryAdam(epoch,1), ...
        DataLoss = lossHistoryAdam(epoch,2), ...
        PhysicsLoss = lossHistoryAdam(epoch,3), ...
        BCLoss = lossHistoryAdam(epoch,4));

    monitor.Progress = 100*epoch/maxnumItAdam;

    if monitor.Stop
        break
    end
    
    % Adam training loop exit conditions - rel residual
    if mod(epoch, 50) == 0 % checking versus residual 50 epochs ago
        relNow  = residualHistoryAdam(epoch, 7);
        relPrev = residualHistoryAdam(epoch - 49, 7);
        if relNow > relPrev * (1 - 0.05) % 5 percent improvement barrier
            fprintf("Adam exit at epoch %d: relRes %.3f %%\n", ...
                epoch, relNow);
            break
        end
    end

end
adamtEnd = toc(adam_tStart);

fprintf("Total time spend on Adam training: %.4f seconds\n\n", adamtEnd);
%% =============== Adam Training Data Saves ===============

% saving Adam data and setting net to best version
adamEpochsRun   = epoch;
lossHistoryAdam     = lossHistoryAdam(1:adamEpochsRun, :);
residualHistoryAdam = residualHistoryAdam(1:adamEpochsRun, :);
net.Learnables = learnables_best;


%% =============== L-BFGS: full-batch data, no minibatchqueue needed ===============
if canUseGPU
    toDL = @(a) dlarray(gpuArray(single(reshape(a, 1, []))), 'CB');
else
    toDL = @(a) dlarray(single(reshape(a, 1, [])), 'CB');
end

% DNS data (all training points)
XdataFull = toDL(XTrain);
YdataFull = toDL(YTrain);
uFull     = toDL(uStarTrain);
vFull     = toDL(vStarTrain);
pFull     = toDL(pStarTrain);

% Physics data (all points)
physXFull = toDL(XphysNet);
physYFull = toDL(YphysNet);
dUUdxFull = toDL(dUUdX);
dUVdyFull = toDL(dUVdY);
dUVdxFull = toDL(dUVdX);
dVVdyFull = toDL(dVVdY);

%% =============== L-BFGS Initialisation ===============

% ---- L-BFGS settings
lbfgsMaxIt        = 3000;
lbfgsHistorySize  = 20;
lbfgsMaxLineSrch  = 30;
gradTol           = 1e-8;     % stop if ||grad|| below this
stepTol           = 1e-10;    % stop if ||step|| below this
diagEvery         = 50;        % epochs between diagnostic evaluations (raise to save time)

% ---- Loss function handle for lbfgsupdate: returns [loss, gradients]
lambdaBCDL   = dlarray(lambdaBC);
lambdaPhysDL = dlarray(lambdaPhys);
lambdaDataDL = dlarray(lambdaData);

accFunLB = dlaccelerate(@modelLoss);
clearCache(accFunLB);
 
lossFcn = @(netIn) dlfeval(accFunLB, netIn, ...
    XdataFull, YdataFull, uFull, vFull, pFull, ...
    physXFull, physYFull, dUUdxFull, dUVdyFull, dUVdxFull, dVVdyFull, scalers.Re, scalers, ...
    bcX, bcY, lambdaPhysDL, lambdaBCDL, lambdaDataDL, B);
 
% lbfgsState only accepts HistorySize, InitialInverseHessianFactor, InitialStepSize
solverState = lbfgsState(HistorySize = lbfgsHistorySize);

lossHistoryLBFGS     = zeros(lbfgsMaxIt, 4);   % [total, data, physics, bc]
residualHistoryLBFGS = zeros(lbfgsMaxIt, 7);   % [contRes, xMomRes, yMomRes, lossU, lossV, lossPres, relRes]
 
monitor.Status = "L-BFGS";     % same monitor object as Adam


%% =============== L-BFGS Training Loop ===============

lbgfstStart = tic;
kEnd = 0;
for k = 1:lbfgsMaxIt
 
    % One L-BFGS step (line search evaluates lossFcn several times internally)
    [net, solverState] = lbfgsupdate(net, lossFcn, solverState, ...
        MaxNumLineSearchIterations = lbfgsMaxLineSrch, ...
        LineSearchMethod           = "weak-wolfe");
 
    kEnd      = k;
    globalIt  = adamEpochsRun + k;          % continues the Adam x-axis
    totalLoss = gather(extractdata(solverState.Loss));
 
    % Diagnostics very 50 iterations
    if mod(k, diagEvery) == 0 || k == 1
        [~, ~, lossData, lossPhys, lossBC, contRes, xMomRes, yMomRes, ...
            lossU, lossV, lossPres, relRes] = dlfeval(accFunLB, net, ...
            XdataFull, YdataFull, uFull, vFull, pFull, ...
            physXFull, physYFull, dUUdxFull, dUVdyFull, dUVdxFull, dVVdyFull, scalers.Re, scalers, ...
            bcX, bcY, lambdaPhysDL, lambdaBCDL, lambdaDataDL, B);
 
        lossHistoryLBFGS(k, :) = gather(extractdata([totalLoss, lossData, lossPhys, lossBC]));
        residualHistoryLBFGS(k, :) = gather(extractdata( ...
            [contRes, xMomRes, yMomRes, lossU, lossV, lossPres, relRes]));
    else
        % carry forward last diagnostics, update total only
        lossHistoryLBFGS(k, :)     = lossHistoryLBFGS(k-1, :);
        lossHistoryLBFGS(k, 1)     = totalLoss;
        residualHistoryLBFGS(k, :) = residualHistoryLBFGS(k-1, :);
    end
 
    % Debug output (same format as Adam)
    if mod(k, 50) == 0 || k == 1
        fprintf("LBFGS %4d | total=%.3e  data=%.3e  physics=%.3e  bc=%.3e\n", ...
            k, lossHistoryLBFGS(k,1), lossHistoryLBFGS(k,2), ...
            lossHistoryLBFGS(k,3), lossHistoryLBFGS(k,4));

        fprintf("  ||grad|| = %.3e  ||step|| = %.3e  lineSearch = %s\n", ...
            gather(extractdata(solverState.GradientsNorm)), ...
            gather(extractdata(solverState.StepNorm)), string(solverState.LineSearchStatus));

        fprintf("Physics | ContRes = %.3e xMomRes = %.3e yMomRes=%.3e \nData | uRes=%.3e  vRes=%.3e pRes=%.3e\nRelative Res = %.2f %% \n\n", ...
            residualHistoryLBFGS(k, 1), residualHistoryLBFGS(k, 2), residualHistoryLBFGS(k, 3), ...
            residualHistoryLBFGS(k, 4), residualHistoryLBFGS(k, 5), residualHistoryLBFGS(k, 6), ...
            residualHistoryLBFGS(k, 7));
    end
 
    % Best-weights tracking (same criterion as Adam)
    if lossHistoryLBFGS(k, 1) < minLoss
        learnables_best = net.Learnables;
        minLoss = lossHistoryLBFGS(k, 1);
    end
 
    % Same monitor as Adam
    updateInfo(monitor, ...
        Epoch = globalIt, ...
        Loss  = lossHistoryLBFGS(k, 1));
    recordMetrics(monitor, globalIt, ...
        TotalLoss   = lossHistoryLBFGS(k, 1), ...
        DataLoss    = lossHistoryLBFGS(k, 2), ...
        PhysicsLoss = lossHistoryLBFGS(k, 3), ...
        BCLoss      = lossHistoryLBFGS(k, 4));
    monitor.Progress = 100 * k / lbfgsMaxIt;
 
    % Exit conditions
    if monitor.Stop
        fprintf("L-BFGS stopped by user at iteration %d.\n", k);
        break
    end
    if solverState.GradientsNorm < gradTol
        fprintf("L-BFGS converged (gradient norm) at iteration %d.\n", k);
        break
    end
    if solverState.StepNorm < stepTol
        fprintf("L-BFGS converged (step norm) at iteration %d.\n", k);
        break
    end
    if solverState.LineSearchStatus == "failed"
        fprintf("L-BFGS line search failed at iteration %d.\n", k);
        break
    end

        % L-BFGS training loop exit conditions - rel residual
    if mod(k, 50) == 0 % checking versus residual 50 epochs ago
        relNow  = residualHistoryLBFGS(k, 7);
        relPrev = residualHistoryLBFGS(k - 49, 7);
        if relNow > relPrev * (1 - 0.01) % 1 percent improvement barrier
            fprintf("L-BFGS exit at iteration %d: relRes %.3f %%\n", ...
                k, relNow);
            break
        end
    end

end
lbgfstEnd = toc(lbgfstStart);

fprintf("Total time spent training using L-BFGS: %.2f seconds\n", lbgfstEnd);

runTimeEnd = toc(runTimeStart);
fprintf("Total Run Time: %.3f seconds\n\n", runTimeEnd);
 
% Trim and merge histories (Adam then L-BFGS) for post-processing
lossHistoryLBFGS     = lossHistoryLBFGS(1:kEnd, :);
residualHistoryLBFGS = residualHistoryLBFGS(1:kEnd, :);
lossHistoryAll       = [lossHistoryAdam; lossHistoryLBFGS];
%residualHistoryAll   = [residualHistory; residualHistoryLBFGS];
 
% Use the best weights found across both phases
net.Learnables = learnables_best;
 

%% =============== Loss Plot Curve ===============
trainingHistoryFig = figure;

semilogy(lossHistoryAll);
legend('total','data','physics','bc');
xlabel('epoch'); ylabel('loss (log scale)');
title('PINN training loss');



%% =============== Validation Calculations ===============
net.Learnables = learnables_best; % setting net to have best weightings
if canUseGPU
    XphysDL = dlarray(gpuArray(single(XphysNet')), "CB");
    YphysDL = dlarray(gpuArray(single(YphysNet')), "CB");
else
    XphysDL = dlarray(XphysNet', "CB");
    YphysDL = dlarray(YphysNet', "CB");
end

[meanAbsDiv, meanAbsTerms, residualDiv] = dlfeval(@continuityResidual, net, scalers, XphysDL, YphysDL, B);
meanAbsDiv = extractdata(meanAbsDiv);
meanAbsTerms = extractdata(meanAbsTerms);
residualDiv = extractdata(residualDiv);

fprintf('mean|div|             = %.3e\n', meanAbsDiv);
fprintf('mean(|du/dx|+|dv/dy|) = %.3e\n', meanAbsTerms);
fprintf('relative residual     = %.2f %% \n ', residualDiv*100);

%% =============== Save Net for Plotting ===============

% Cannot save net as is, need to save its learnables and hyperparameters
% and use buildNetwork function to rebuild it in another script (plotting
% script)

% Also need to gather all the saving parameters so when realled they are
% not gpuArray
learnables_best.Value = cellfun(@gather, learnables_best.Value, 'UniformOutput', false);
if isa(B, 'gpuArray'), B = gather(B); end

save('trainedPINN.mat', 'learnables_best', 'scalers', 'inputDim', 'outputDim', 'layerSize', 'numBlocks', 'mu', 'sigma', 'B', 'valData');

outputDir = "PINN_Figures";
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end


%% =============== Saving Training Figures ===============
exportgraphics(trainingHistoryFig, fullfile(outputDir, 'trainingHistory.png') , 'Resolution', 200);
savefig(trainingHistoryFig,  fullfile(outputDir, 'trainingHistory.fig'));
