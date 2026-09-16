function lossBC = wallLoss(net, Xwall, Ywall, B)
    % net:      dlnetwork, 2 inputs 3 outputs
    % Xwall, Ywall:     X and Y BC co-ords
    
    XYwall = cat(1, Xwall, Ywall);
    XYwallenc = dlarray(fourierFeatures(stripdims(XYwall), B), "CB");
    pred = forward(net, XYwallenc);

    uWall = pred(1,:);
    vWall = pred(2,:);
    % no p wall boundary condition
    
    % BC: u,v @ wall = 0
    lossBC = mean(uWall.^2, 'all') + mean(vWall.^2, 'all');
end
