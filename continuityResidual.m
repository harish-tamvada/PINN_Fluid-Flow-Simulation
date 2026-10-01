function [meanAbsDiv, meanAbsTerms, residualDiv] = continuityResidual(net, scalers, X, Y, B)
% continuity residual dudx + dvdy
% run the x,y mesh grid through net and use calculated u,v values

XY = cat(1, X, Y);
XYenc = dlarray(fourierFeatures(stripdims(XY), B), "CB");
pred = predict(net, XYenc);
uPred = pred(1,:);
vPred = pred(2,:);

cX = (2.0 / scalers.xmax) * (1.0 / scalers.H);
cY = (2.0 / scalers.ymax) * (1.0 / scalers.H);

du_dX = dlgradient(sum(uPred, 'all'), X, 'EnableHigherDerivatives', true);
dv_dY = dlgradient(sum(vPred, 'all'), Y, 'EnableHigherDerivatives', true);

du_dx = du_dX * cX;
dv_dy = dv_dY * cY;

div = du_dx + dv_dy;
meanAbsDiv = mean(abs(div(:)));
meanAbsTerms = mean(abs(du_dx(:)) + abs(dv_dy(:)));
residualDiv = meanAbsDiv / meanAbsTerms;
end