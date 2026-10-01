function data = loadData(folder, cases, fields_data, fields_labels)

data = struct();

for i = numel(cases)

    c = cases{i};
    cKey = matlab.lang.makeValidName(c);

    for j = 1:numel(fields_data)
        
        f = fields_data{j};
        fKey = matlab.lang.makeValidName(f);

        fname = fullfile(folder, "komegasst_" + c + "_" + f + ".mat");

        if ~isfile(fname)
            warning("Missing file: %s", fname);
            continue
        end
        
        S = load(fname);

        vars = fieldnames(S);
        if isfield(S, f)
            data.(cKey).(fKey) = S.(f);        % variable named like the field

        elseif isscalar(vars)
            data.(cKey).(fKey) = S.(vars{1});  % only one variable in the file

        else
            data.(cKey).(fKey) = S;            % keep everything if ambiguous
        end
        
    end
    
    for l = 1:numel(fields_labels)
        
        f = fields_labels{l};
        lKey = matlab.lang.makeValidName(f);

        fname = fullfile(folder, c + "_" + f + ".mat");

        if ~isfile(fname)
            warning("Missing file: %s", fname);
            continue
        end
        
        S = load(fname);

        vars = fieldnames(S);
        if isfield(S, f)
            data.(cKey).(lKey) = S.(f);        % variable named like the field

        elseif isscalar(vars)
            data.(cKey).(lKey) = S.(vars{1});  % only one variable in the file

        else
            data.(cKey).(lKey) = S;            % keep everything if ambiguous
        end
        
    end

end






end