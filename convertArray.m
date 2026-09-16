function deepLarray = convertArray(X)

    if canUseGPU
        deepLarray = dlarray(gpuArray(single(X)), "CB");
    else
        deepLarray = dlarray(X, "CB");
    end
    
end