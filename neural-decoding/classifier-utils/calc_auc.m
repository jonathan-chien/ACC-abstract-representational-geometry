function [aucroc,aucpr] = calc_auc(groundTruth,scores,nvp)
% Calculate AUC for each class (that is, letting each class be the positive
% class) based on both the ROC curve and the Precision-Recall (PR) curve.
% Note that AUC ROC is generally sensitive to algorithms biased toward the
% majority class, but under extreme class rarity, it may be unreliable, and
% AUC PR may be preferrable.
%
% PARAMETERS
% ----------
% groundTruth -- n-vector of ground truth labels over observations, where n
%                = number of observations.
% scores      -- nObs x 2 matrix of scores, where the i_th j_th element
%                contains the score for classifying observation i into
%                class j.
% Name-Value Pairs (nvp)
%   'handleNaN' -- (1 (default) | 0), specify the behavior if NaN values
%                  are detected among the precision or recall values at
%                  some threshold. If 0, the NaN values are left as is,
%                  causing MATLAB to ignore those values when calculating
%                  AUC. If there is a large gap between the last non-NaN
%                  value and 1, the area might be calculated under a much
%                  smaller portion of the curve. If this is not desirable,
%                  set 'handleNaN' to true; in this case, the false
%                  negative rate (FNR) and false positive rate (FPR) will
%                  be checked for thresholds resulting in NaN recall and
%                  precision values, respectively. If FNR and/or FPR are 0,
%                  the recall and/or precision (respectively) at that
%                  threshold will be reassigned to 1. If this reassignment
%                  is successful for all NaN values, the AUCPR will be
%                  recalculated manually on the corrected curves using
%                  trapezoidal approximation (as does perfcurve.m). If any
%                  of the attempted reassignments are unsuccessful, the
%                  AUCPR will be left as the default value calculated by
%                  MATLAB (where NaN values in the curves are dropped).
% 
% RETURNS
% -------
% aucroc -- c-vector of AUC ROC values, where c = nClasses.
% aucpr  -- c-vecotr of AUC PR values, where c = nClasses.
%
% Author: Jonathan Chien.

arguments
    groundTruth
    scores
    nvp.handleNaN = false
end

% Get class indices and number of classes.
classIdc = unique(groundTruth);
nClasses = length(classIdc);  

% Calculate AUC based on both ROC and Precision-Recall (PR) curves.
aucroc = NaN(nClasses, 1);
aucpr = NaN(nClasses, 1);
for iClass = 1:nClasses

    % Calculate tpr, fpr, precision, recall vectors, as well as areas under
    % curve.
    [fpr, tpr, ~, aucroc(iClass)] ...
        = perfcurve(groundTruth, scores(:,iClass), classIdc(iClass));
    [rec, pre, ~, aucpr(iClass)] ...
        = perfcurve(groundTruth, scores(:,iClass), classIdc(iClass), ...
                    'XCrit', 'reca', 'YCrit', 'prec');

    % Check for NaN precision/recall values and handle them.
    if nvp.handleNaN
        nanRecIdc = find(isnan(rec));
        nanPreIdc = find(isnan(pre));
        
        % Handle NaN recall values.
        if nanRecIdc
            for iProb = nanRecIdc
                if ~any([tpr(iProb) fpr(iProb) (1 - tpr(iProb))]) 
                    rec(iProb) = 1;
                elseif tpr(iProb) == 0
                    rec(iProb) = 0;
                end
            end
        end
        
        % Handle NaN precision values.
        if nanPreIdc 
            for iProb = nanPreIdc
                if ~any([tpr(iProb) fpr(iProb) (1 - tpr(iProb))]) 
                    pre(iProb) = 1;
                elseif tpr(iProb) == 0
                    pre(iProb) = 0;
                end
            end
        end

        % Calculate AUC-PR using trapezoidal approximation (as does
        % perfcurve.m) if NaNs were detected and replacement was
        % successful.
        if (~isempty(nanRecIdc) || ~isempty(nanPreIdc)) 
            aucpr(iClass) = trapezoidal_approximation(rec, pre);
        end
    end
    
end

end


% --------------------------------------------------
function auc = trapezoidal_approximation(x,y)
% Calculate the area under a curve for x vs y values using trapezoidal
% approximation.
    heights = diff(x); if isrow(heights), heights = heights'; end
    bases(:,1) = y(1:end-1);
    bases(:,2) = y(2:end);
    auc = 0.5 * sum(heights .* sum(bases, 2));
end
