clc; clear

%% Loading the net, rebuilding it from .mat

trained = load('trainedPINN.mat');

%Rebuild net
net = buildNetwork(trained.inputDim, trained.outputDim, trained.layerSize, trained.numBlocks, trained.mu, trained.sigma);
net.Learnables = trained.learnables;   

%extract data etc
data = trained.data;
scalers = trained.scalers;
B = trained.B;

%dir for figures
outputDir = "PINN_Figures";

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end


%% ---- Prepare data ----

%X Y ready for training
X = 2 * (data.Cx / scalers.H) / scalers.xmax  - 1;
Y = 2 * (data.Cy / scalers.H) / scalers.ymax  - 1;

X = convertArray(X);
Y = convertArray(Y);

%Ux Uy p uu uv vv ready for training
Ub2 = scalers.Ub^2;
uStar  = data.Ux / scalers.Ub;
vStar  = data.Uy / scalers.Ub;
pStar  = data.p  / Ub2;

XY = cat(1, X, Y);
XYenc = dlarray(fourierFeatures(stripdims(XY), B), "CB");
pred = predict(net, XYenc);
pred = double(gather(extractdata(pred)));

uPred = pred(1,:);
vPred = pred(2,:);
pPred = pred(3,:);

xStar = data.Cx / scalers.H;
yStar = data.Cy / scalers.H;

%% ---- 1. Scatter: predicted vs. true ----
figScatter = figure('Name', 'Predicted vs. True', 'Position', [100 100 1200 400]);

plotScatterComparison(1, uStar, uPred, 'u*');
plotScatterComparison(2, vStar, vPred, 'v*');
plotScatterComparison(3, pStar, pPred, 'p*');

saveFigure(figScatter, outputDir, 'scatter_comparison');

%% ---- 2. Contour maps: DNS vs. prediction vs. error ----
figContourU = plotContourComparison(xStar, yStar, uStar, uPred, 'u*');
saveFigure(figContourU, outputDir, 'contour_u');

figContourV = plotContourComparison(xStar, yStar, vStar, vPred, 'v*');
saveFigure(figContourV, outputDir, 'contour_v');

figContourP = plotContourComparison(xStar, yStar, pStar, pPred, 'p*');
saveFigure(figContourP, outputDir, 'contour_p');

%% ---- 3. Streamwise profiles at fixed x* locations ----
xLocations = 0:1:8;
figProfile = plotProfileComparison(xStar, yStar, uStar, uPred, xLocations, 'u*');
saveFigure(figProfile, outputDir, 'profile_u');

fprintf('Saved 5 figures to "%s"\n', outputDir);

 
%% ===== local helper functions =====
 
function saveFigure(fig, outputDir, name)
    % Makes the saving look cleaner
    exportgraphics(fig, fullfile(outputDir, [name '.png']), 'Resolution', 200);
    savefig(fig, fullfile(outputDir, [name '.fig']));
end

%% ===== local helper functions =====

function plotScatterComparison(subplotIdx, trueVal, predVal, fieldName)
    subplot(1, 3, subplotIdx);
    % Plotting DNS data vs net predict data
    scatter(trueVal, predVal, 8, 'filled', 'MarkerFaceAlpha', 0.3);
    hold on;

    lims = [min([trueVal,predVal]), max([trueVal,predVal])];
    plot(lims, lims, 'r--', 'LineWidth', 1.5);
    hold off;
    axis equal; grid on;
    xlim(lims); ylim(lims);
    xlabel(['DNS ' fieldName]); ylabel(['Predicted ' fieldName]);
    
    % R^2, RMSE calc + addition to the plot title
    ssRes = sum((trueVal - predVal).^2);
    ssTot = sum((trueVal - mean(trueVal)).^2);
    r2 = 1 - ssRes/ssTot;
    rmse = sqrt(mean((trueVal - predVal).^2));
    title(sprintf('%s: R^2=%.4f, RMSE=%.2e', fieldName, r2, rmse));
end


function fig = plotContourComparison(xStar, yStar, trueVal, predVal, fieldName)
    xStar = xStar(:); yStar = yStar(:);
    trueVal = trueVal(:); predVal = predVal(:);
 
    % Interpolates the scattered data onto a regular grid for contourf.
    % griddata returns NaN outside the convex hull of the scattered
    % points, which avoids plotting inside the solid hill.
    xg = linspace(min(xStar), max(xStar), 300);
    yg = linspace(min(yStar), max(yStar), 150);
    [Xg, Yg] = meshgrid(xg, yg);
 
    trueGrid = griddata(xStar, yStar, trueVal, Xg, Yg, 'natural');
    predGrid = griddata(xStar, yStar, predVal, Xg, Yg, 'natural');
    errGrid  = abs(trueGrid - predGrid);
 
    fig = figure('Name', ['Contours: ' fieldName], 'Position', [100 100 1200 600]);
 
    valRange = [min(trueVal), max(trueVal)];
 
    subplot(3,1,1);
    contourf(Xg, Yg, trueGrid, 40, 'LineStyle', 'none');
    colorbar; clim(valRange); axis equal tight;
    title(['DNS ' fieldName]);
 
    subplot(3,1,2);
    contourf(Xg, Yg, predGrid, 40, 'LineStyle', 'none');
    colorbar; clim(valRange); axis equal tight;
    title(['Predicted ' fieldName]);
 
    subplot(3,1,3);
    contourf(Xg, Yg, errGrid, 40, 'LineStyle', 'none');
    colorbar; axis equal tight;
    title(['Absolute error |DNS - Predicted| ' fieldName]);
    xlabel('x*');
end


function fig = plotProfileComparison(xStar, yStar, trueVal, predVal, xLocations, fieldName)
    xStar = xStar(:); yStar = yStar(:);
    trueVal = trueVal(:); predVal = predVal(:);
 
    fig = figure('Name', ['Profiles: ' fieldName], 'Position', [100 100 1400 400]);
    numLoc = numel(xLocations);
    tol = 0.15;   % half-width of the x* slice band, in units of H
 
    for i = 1:numLoc
        x0 = xLocations(i);
        mask = abs(xStar - x0) < tol;
 
        if ~any(mask)
            continue;
        end
 
        yLocal = yStar(mask);
        trueLocal = trueVal(mask);
        predLocal = predVal(mask);
 
        [yLocalSorted, sortIdx] = sort(yLocal);
 
        subplot(1, numLoc, i);
        plot(trueLocal(sortIdx), yLocalSorted, 'k-', 'LineWidth', 1.5); hold on;
        plot(predLocal(sortIdx), yLocalSorted, 'r--', 'LineWidth', 1.5); hold off;
        title(sprintf('x*=%d', x0));
        if i == 1
            ylabel('y*');
            legend('DNS', 'Predicted', 'Location', 'best');
        end
        xlabel(fieldName);
        grid on;
    end
    sgtitle(['Streamwise profile comparison: ' fieldName]);
end