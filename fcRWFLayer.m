classdef fcRWFLayer < nnet.layer.Layer

    properties (Learnable)
        S      % scale vector, [outputSize x 1]
        V      % weight matrix, [outputSize x inputSize]
        Bias   % [outputSize x 1]
    end

    methods
        function layer = fcRWFLayer(outputSize, inputSize, mu, sigma, name)
            layer.Name = name;
            layer.Description = "RWF fully connected, " + outputSize + " units";

            % --- Initialize V using Glorot scheme ---
            bound = sqrt(6/(inputSize + outputSize));
            layer.V = single((2*bound)*rand(outputSize, inputSize) - bound);

            % --- Initialize scale factor s ~ N(mu, sigma^2) ---
            layer.S = single(mu + sigma*randn(outputSize, 1));

            % --- Bias initialized to zero ---
            layer.Bias = single(zeros(outputSize, 1));
        end

        function Z = predict(layer, X)
            % Reconstruct W = diag(exp(s)) * V
            W = exp(layer.S) .* layer.V;   % row-wise scaling, broadcasts correctly
            Z = W * X + layer.Bias;
        end
    end
end