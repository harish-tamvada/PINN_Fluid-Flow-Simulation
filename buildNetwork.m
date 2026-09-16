function net = buildNetwork(inputDim, outputDim, layerSize, numBlocks, mu, sigma)

    % Input layer
    layers = featureInputLayer(inputDim, 'Normalization','none');
    
    % Hidden Layers
    prevSize = inputDim;
    for i = 1:numBlocks
        layers = [layers
            fcRWFLayer(layerSize, prevSize, mu, sigma, "rwf_fc " + i)
            tanhLayer("Name", "tanh " + i)
            ];
        prevSize = layerSize;
    end
            
    % Output Layer (prints to command line the network)
    layers = [layers
        fcRWFLayer(outputDim, layerSize, mu, sigma, "rwf_output")
        ]
    
    
    net = dlnetwork(layers)
    
    % Convert net to allow gpu usage (single data type and gpuarray)
    net = dlupdate(@single, net);
    
    if canUseGPU
        disp("GPU available: " + gpuDevice().Name);
        net = dlupdate(@gpuArray, net);
    end

end