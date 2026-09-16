function encoded = fourierFeatures(XYT, B)
    % XYT: stacked inputs [2 * N]
    % B  : Encoder        [ff * 2]
    projection = B * XYT;

    % Output [2ff * N] encoded matrix
    encoded = [cos(projection); sin(projection)];
end