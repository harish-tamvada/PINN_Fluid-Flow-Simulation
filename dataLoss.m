function lossData = dataLoss(net, X, Y, uTarget, vTarget, pTarget, B)

    % net:      dlnetwork, 2 inputs 3 outputs
    % X, Y:     X,Y in network [-1,1] coords
    % uTarget, vTarget, pTarget: DNS target data non-dimensionalised

    XY = cat(1, X, Y);
    XYenc = dlarray(fourierFeatures(stripdims(XY), B), "CB");
    pred = forward(net, XYenc);

    uPred = pred(1,:);
    vPred = pred(2,:);
    pPred = pred(3,:);

    lossU = mean((uPred - uTarget).^2, 'all');
    lossV = mean((vPred - vTarget).^2, 'all');
    lossP = mean((pPred - pTarget).^2, 'all');
    

    % Found v to be noisy and difficult to train so increased weight
    lossData = lossU + 1.5 * lossV + lossP;


end
