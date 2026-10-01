function varargout = rowRescaleFunc(varargin)
% Reshapes each read-in variable of a minibatch into a 1-by-batchSize row
% vector, matching the "CB" (1 channel, batchSize observations) layout that
% convertArray previously produced. Without this, minibatchqueue's default
% stacking leaves data as batchSize-by-1 columns, which transposes the
% inputs to fourierFeatures/dataLoss and breaks the B * XYT multiplication.
varargout = cell(1, nargin);
for i = 1:nargin
    varargout{i} = reshape(cat(1, varargin{i}{:}), 1, []);
end
end