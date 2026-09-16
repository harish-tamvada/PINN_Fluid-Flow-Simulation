function [loss, gradients, lossD, lossP, lossB] = modelLoss(net, ...
        Xdata, Ydata, uTarget, vTarget, pTarget, ...
        Xphys, Yphys, dUUdx, dUVdy, dUVdx, dVVdy, Re, scalers, ...
        Xwall, Ywall, lambdaPhys, lambdaBC, B)
    
    % Overall loss function - calls 3 individual loss functions and adds
    % the together
    
    % Temporal weights could be added here instead of fixed
    % lambdaPhys/laambdaBC in the future

    lossD = dataLoss(net, Xdata, Ydata, uTarget, vTarget, pTarget, B);
    lossP = physicsResidualLoss(net, Xphys, Yphys, dUUdx, dUVdy, dUVdx, dVVdy, Re, scalers, B);
    lossB = wallLoss(net, Xwall, Ywall, B);

    loss = lossD + lambdaPhys * lossP + lambdaBC * lossB;

    gradients = dlgradient(loss, net.Learnables);
end
