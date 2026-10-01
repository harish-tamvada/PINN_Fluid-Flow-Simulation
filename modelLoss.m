function [loss, gradients, lossD, lossP, lossB, contRes, xMomRes, yMomRes, lossU, lossV, lossPres, relRes] = modelLoss(net, ...
        Xdata, Ydata, uTarget, vTarget, pTarget, ...
        Xphys, Yphys, dUUdx, dUVdy, dUVdx, dVVdy, Re, scalers, ...
        Xwall, Ywall, lambdaPhys, lambdaBC, lambdaData, B)
    
    % Overall loss function - calls 3 individual loss functions and adds
    % the together
    
    % Temporal weights could be added here instead of fixed
    % lambdaPhys/laambdaBC in the future

    [lossD, lossU, lossV, lossPres] = dataLoss(net, Xdata, Ydata, uTarget, vTarget, pTarget, B);
    [lossP, contRes, xMomRes, yMomRes, relRes] = physicsResidualLoss(net, Xphys, Yphys, dUUdx, dUVdy, dUVdx, dVVdy, Re, scalers, B);
    lossB = wallLoss(net, Xwall, Ywall, B);

    loss = lambdaData * lossD + lambdaPhys * lossP + lambdaBC * lossB;

    gradients = dlgradient(loss, net.Learnables);
end
