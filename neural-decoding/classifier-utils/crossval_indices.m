function indices = crossval_indices(nObs,kFold,nvp)
% Returns a vector of fold indices whose elements are integers from the
% set {1, 2, ..., k}, where k = the number of folds to be used in
% cross-validation. The number of observations in each fold will be as
% balanced as possible, though if the total number of observations is not
% evenly divisible by the number of desired folds, the extra observations
% will be assigned beginning with those folds whose indices come first
% ordinally.
% 
% PARAMETERS
% ----------
% nObs  -- Positive integer equal to the number of total observations.
% kFold -- Positive integer equal to the number of desired cross-validation
%          folds.
% Name-Value Pairs (nvp)
%   'permute' -- (1 (default) | 0), if true, the indices will be randomly
%                permuted before being returned. If false, the indices will
%                cycle through the set {1, 2, ... k} in order such that the
%                n*k+1_th element begins anew at 1 for positive integers n.
%
% RETURNS
% -------
% indices -- Vector of cross-validation fold indices equal in length to
%            the number of observations. The i_th element is a fold index
%            that is a member of the set {1, 2, ..., k}, such that the i_th
%            observation is assigned to the fold with this index.
%       
% Author: Jonathan Chien.

arguments
    nObs
    kFold
    nvp.permute = true
end

% Preallocate.
indices = NaN(nObs, 1);

% Assign each observation to a fold.
iFold = 0;
for iObs = 1:nObs
    if iFold < kFold
        iFold = iFold + 1;
    else
        iFold = 1;
    end
    
    indices(iObs) = iFold;
end

% Option to permute indices.
if nvp.permute, indices = indices(randperm(nObs)); end

end
