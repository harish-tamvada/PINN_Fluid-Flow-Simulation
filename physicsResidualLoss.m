function [lossPhysics, contRes, xMomRes, yMomRes] = physicsResidualLoss(net, X, Y, dUUdx, dUVdy, dUVdx, dVVdy, Re, scalers, B)
    % Steady, non-dimensional incompressible RANS residual:
    
    %   continuity:   du*/dx* + dv*/dy* = 0
    %   x-momentum:   u* du*/dx* + v* du*/dy* = -dp*/dx* + (1/Re) lap(u*) - d(uu*)/dx* - d(uv*)/dy*
    %   y-momentum:   u* dv*/dx* + v* dv*/dy* = -dp*/dy* + (1/Re) lap(v*) - d(uv*)/dx* - d(vv*)/dy*
    
    % Inputs:
    % net:      dlnetwork, 2 inputs 3 outputs
    % X, Y:     filtered x,y meshgrid

    %   dUUdx, dUVdy,
    %   dUVdx, dVVdy     numeric Reynolds-stress gradient terms calculated
    %                    within PHLL_PINN.m

    %   Re               scalers.Re

    %   scalers          struct built in PHLL_PINN.m
    %
    % Returns the combined scalar loss plus the three raw residual arrays
    % used for diagnostical purposes
    
    invertRe = 1.0 / Re;

    % constant chain-rule scale factors: network coords (X,Y in [-1,1])
    % -> physical non-dim coords (x*,y* in units of H). Precomputed once,
    % not part of the autodiff graph.
    cX = (2.0 / scalers.xmax) * (1.0 / scalers.H);
    cY = (2.0 / scalers.ymax) * (1.0 / scalers.H);

    % ---- forward pass ----
    XY = cat(1, X, Y);
    XYenc = dlarray(fourierFeatures(stripdims(XY), B), "CB");
    UVP = forward(net, XYenc);
    u = UVP(1,:);
    v = UVP(2,:);
    p = UVP(3,:);

    % ---- first derivatives w.r.t. network coords ----
    % 'EnableHigherDerivatives' + 'RetainData' on the u,v branches since we
    % differentiate du_dX, du_dY, dv_dX, dv_dY a second time below for the
    % viscous Laplacian term. Not needed for p (only used to first order).
    du_dX = dlgradient(sum(u,'all'), X, 'EnableHigherDerivatives', true, 'RetainData', true);
    du_dY = dlgradient(sum(u,'all'), Y, 'EnableHigherDerivatives', true, 'RetainData', true);
    dv_dX = dlgradient(sum(v,'all'), X, 'EnableHigherDerivatives', true, 'RetainData', true);
    dv_dY = dlgradient(sum(v,'all'), Y, 'EnableHigherDerivatives', true, 'RetainData', true);
    dp_dX = dlgradient(sum(p,'all'), X, 'RetainData', true);
    dp_dY = dlgradient(sum(p,'all'), Y, 'RetainData', true);

    % ---- second derivatives w.r.t. network coords (for the Laplacian) ----
    d2u_dX2 = dlgradient(sum(du_dX,'all'), X, 'RetainData', true);
    d2u_dY2 = dlgradient(sum(du_dY,'all'), Y, 'RetainData', true);
    d2v_dX2 = dlgradient(sum(dv_dX,'all'), X, 'RetainData', true);
    d2v_dY2 = dlgradient(sum(dv_dY,'all'), Y, 'RetainData', true);

    % ---- chain rule: network-coord derivatives -> physical non-dim derivatives ----
    % mapping is affine and separable (X depends only on x*, Y only on y*),
    % so there are no cross terms -- each direction just gets a constant scale.
    du_dx = du_dX * cX;   du_dy = du_dY * cY;
    dv_dx = dv_dX * cX;   dv_dy = dv_dY * cY;
    dp_dx = dp_dX * cX;   dp_dy = dp_dY * cY;

    lap_u = d2u_dX2 * (cX^2) + d2u_dY2 * (cY^2);
    lap_v = d2v_dX2 * (cX^2) + d2v_dY2 * (cY^2);

    % ---- assemble residuals ----
    % dUUdx etc. are plain numeric row vectors -- MATLAB implicitly combines
    % them with the dlarray terms, and since they don't depend on network
    % weights, no gradient flows into them (which is correct: they're data).
    contRes = du_dx + dv_dy;

    xMomRes = u.*du_dx + v.*du_dy + dp_dx - invertRe*lap_u + dUUdx(:)' + dUVdy(:)';
    yMomRes = u.*dv_dx + v.*dv_dy + dp_dy - invertRe*lap_v + dUVdx(:)' + dVVdy(:)';

    lossPhysics = mean(contRes.^2, 'all') + mean(xMomRes.^2, 'all') + mean(yMomRes.^2, 'all');
end

