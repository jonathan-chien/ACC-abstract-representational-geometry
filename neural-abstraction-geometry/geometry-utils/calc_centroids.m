function C = calc_centroids(T,nConds)
% Calculate condition cluster centroids by averaging across all trials
% within each condition. 
%
% PARAMETERS
% ----------
% T      : nTrials x nNeurons matrix of firing rates.
% nConds : Scalar number of conditions (among the trials).
% 
% RETURNS
% -------
% C : nConds x nNeurons of firing rates (corresponding to cluster
%     centroids).
%
% Author: Jonathan Chien 


% Determine number of neurons, trials, and trials per condition.
[nTrials, nNeurons] = size(T);
nTrialsPerCond = nTrials / nConds;
assert(mod(nTrials, nConds) == 0, 'nTrials must be evenly divisible by nConds.')

% Average within condition for all conditions.
C = NaN(nConds, nNeurons);
for iCond = 1:nConds
    C(iCond,:) ...
        = mean(T((iCond-1)*nTrialsPerCond + 1 : iCond*nTrialsPerCond, :));
end

end