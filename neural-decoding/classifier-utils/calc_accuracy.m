function [accuracy,balAccuracy] = calc_accuracy(confusionMat)
% Calculate multi-class (including binary) accuracy = micro-precision,
% micro-recall, micro-fmeasure, as well as balanced accuracy (arithmetic
% mean of recall across classes, where each class has equal weight
% regardless of class size).
% 
% PARAMETERS
% ----------
% confusionMat -- nClasses x nClasses confusion matrix.
%
% RETURNS
% -------
% accuracy    -- Accuray as a proportion, between 0 and 1, of true
%                positives (for each class) over total number of
%                observations.
% balAccuracy -- Balanced accuracy, the arithmetic mean of recall across
%                classes, where each class has equal weight, regardless of
%                class size, to prevent bias toward larger classes.
%
% Author: Jonathan Chien.

accuracy = trace(confusionMat) / sum(confusionMat, 'all');

balAccuracy = mean(diag(confusionMat) ./ sum(confusionMat, 2));

end
