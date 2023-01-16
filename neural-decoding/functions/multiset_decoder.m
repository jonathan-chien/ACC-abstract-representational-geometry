function performance = multiset_decoder(T,groundTruth,nvp)
% Wrapper function accepting as inputs multiple predictor matrices and/or
% sets of class labels. For example, an m x n x p array of features could
% correspond to m bins, n single trials, and p neurons. For each n x p
% slice of the input array, a classifier model will be evaluated via k-fold
% cross-validation. Another possible use is to pass in a single n x p
% predictor matrix, along with a matrix of n x m class labels (where the
% i_th column is the i_th set out of m sets of n labels). If desired,
% significance measures such as p values will be computed via permutation.
%
% PARAMETERS
% ----------
% T           : Either an m x n x p array, where m = number of datasets
%               (e.g., bins), n = number of observations (e.g. single
%               trials), and p = number of features (e.g., neurons), or an
%               n x p array (with n and p defined the same). The i_th j_th
%               k_th element would then be the firing rate of the k_th
%               neuron on the j_th single trial in the i_th bin (or, in the
%               case of 2D array inputs, the i_th j_th element is the
%               firing rate of the j_th neuron on the i_th single trial).
%               Note that if T is 3D and groundTruth (see below) is 2D,
%               size(T, 1) must equal size(groundTruth, 2); that is, the
%               number of datsets must match. Otherwise, T is 3D while
%               groundTruth is a vector of length n, or T is 2D (n x p)
%               while groundTruth is a matrix of size n x m. If T is passed
%               in as 2D and groundTruth is a vector, the function will
%               suggest that the user call classifier_significance.m
%               directly.
% groundTruth : Either a vector of length n or a matrix of size n x m. The
%               i_th column is a vector of integer class labels; each label
%               corresponds to some class recognized by the classifier, not
%               necessarily a task condition. See notes about matching T
%               array above.
% Name-Value Pairs (nvp)
%   'dropInd'         : ([] (default) | q-vector). Specify a vector of
%                       length q where q is the number of neurons (out of
%                       the p total) to be dropped for decoding across all
%                       bins/datasets. Elements correspond to indices of
%                       the dropped neurons in the overall population.
%   'classifier'      : (@fitclinear (binary) @fitcecoc (multiclass) |
%                       function handle). Provide function handle of the
%                       classifier to be used. Default is @fitclinear for
%                       binary classification and @fitcecoc for nClasses
%                       > 2.
%   'classCoding'     : String value corresponding to acceptable values
%                       for fitcecoc.m's 'Coding' name-value pair. Sets
%                       coding design for classifying more than 2 classes.
%                       If 'classifier' is not @fitcecoc, this argument is
%                       ignored. Default = 'onevsall'.
%   'cvFun'           : (1 (default) | 2), specify whether to use
%                       crossval_model_1 or crossval_model_2 to evaluate
%                       model.
%   'kFold'           : (10 (default) | Positive integer value). Specify 
%                       the number of partitions/folds for cross-validation.
%   'nCvReps'         : (50 | positive integer). Number of repetitions of
%                       cross-validation process (performance metrics are
%                       calculated for each repetition and then averaged)
%                       for the unpermuted data only. Only one repetition
%                       is performed for each permuted dataset.
%   'useGpu'          : (1 | 0 (default)), specify whether or not to
%                       convert predictor matrix to GPU array. Note that
%                       GPU arrays are not supported by all MATLAB
%                       classifier functions.
%   'oversampleTrain' : (string | false (default)), specify whether or not
%                       to oversample train set. If so, specify either
%                       'byClass' to oversample within classes or 'byCond'
%                       to oversample within conditions (see 'condLabels'
%                       below). Set false to suppress oversampling.
%   'oversampleTest'  : (string | false (default)), specify whether or not
%                       to oversample test set. If so, specify either
%                       'byClass' to oversample within classes or 'byCond'
%                       to oversample within conditions (see 'condLabels'
%                       below). Set false to suppres oversampling.
%   'condLabels'      : ([] (default) | n_observations x 1 vector). If
%                       vector, must have elements in one-to-one
%                       correspondence to both the rows of X and the
%                       elements of ground_truth. The i_th element is the
%                       condition index of the i_th observation/trial.
%   'nResamples'      : (100 (default) | positive integer). Number of
%                       resamples to take (either within each class or
%                       within each condition, see 'oversampleTrain' and
%                       'oversampleTest' above).
%   'pval'            : ('right-tailed' (default) | 'left-tailed' |
%                       'two-tailed' | false). Specify the sidedness of the
%                       test. If false, no permutations will be performed
%                       (resulting in no signfiicance measures of any
%                       kind); it may be useful to suppress these
%                       permutations if this function is called by another
%                       function, and significance measures are not desired
%                       at that time.
%   'nullInt'         : (95 (default) | scalar in [0,100]). Specify the
%                       null distribution interval size to be returned.
%   'nPerms'          : (1000 (default) | positive integer). Specify the
%                       number of random permutations (each resulting in
%                       one sample from the null distribution).
%   'permute'         : ('features' | 'labels' (default)), string value
%                       specifying how to generate each dataset (one
%                       permutation). If 'features', each column of the
%                       predictor matrix (vector across observations for
%                       one feature) is permuted independently; this
%                       essentially shuffles the labels with respect to the
%                       observations, independently for each feature. If
%                       'labels', the labels themselves are permuted once
%                       (leaving intact any correlations among features).
%   'saveLabels'      : (1 | 0 (default)), specify whether or not to save
%                       the predicted labels (for the original unpermuted
%                       data only). Note that labels cannot be returned if
%                       'oversampleTest' evaluates to true.
%   'saveScores'      : (1 | 0 (default)), specify whether or not to save
%                       the predicted scores (for the original unpermuted
%                       data only). Note that scores cannot be returned if
%                       'oversampleTest' evaluates to true.
%   'saveConfMat'     : (1 | 0 (default)), specify whether or not to save
%                       the confusion matrices (for each of the reptitions
%                       on the original unpermuted data only).
%   'concatenate'     : (1 (default) | 0), if true, results from each bin
%                       will be concatenated in a single numeric array
%                       under the corresponding field name. If false,
%                       performance will be returned as an nBins x 1 cell
%                       array, where each cell contains the output of
%                       classifier_signficance for the corresponding bin.
%
% RETURNS
% -------
% performance : If 'concatenate' = false, an nBins x 1 cell array, where
%               the i_th cell contains the output of
%               classifier_significance.m (see RETURNS under that
%               function's documentation) for the i_th bin. If
%               'concatenate' = true, performance is a scalar struct with
%               the following fields:
%   .accuracy    : nBins x 1 vector whose i_th element is the
%                  micro-averaged accuracy across classes (for an
%                  aggregation across folds, this is the sum of true
%                  positive for each class, divided by the total number of
%                  observations; this is also equivalent to micro-averaged
%                  precision, recall, and f-measure, since each observation
%                  is assigned to one and only one class), averaged across
%                  repetitions.
%   .balaccuracy : nBins x 1 vector whose i_th element is the balanced
%                  accuracy (unweighted arithmetic mean of recall across
%                  classes). For binary classififcation, the recall of the
%                  two classes is also called senstivity and specificity.
%   .precision   : nBins x nClasses matrix whose i_th j_th element is the
%                  precision (number of true positives divided by number of
%                  predicted positives) for the j_th class as positive,
%                  averaged across repetitions, in the i_th bin.
%   .recall      : nBins x nClasses matrix whose i_th j_th element is the
%                  precision (number of true positives divided by number of
%                  actual positives) for the j_th class as positive,
%                  averaged across repetitions, in the i_th bin.
%   .fmeasure    : nBins x nClasses matrix of f-measure scores whose i_th
%                  j_th element is the f-measure for the j_th class as
%                  positive, averaged across repetitions, in the i_th bin.
%   .aucroc      : nBins x nClasses matrix of AUC ROC values whose i_th
%                  j_th element is the AUC ROC for the j_th class as
%                  positive, averaged across repetitions, in the i_th bin.
%   .aucpr       : nBins x nClasses matrix of AUC PR (precision-recall)
%                  values whose i_th j_th element is the AUC PR for the
%                  j_th class as positive, averaged across repetitions, in
%                  the i_th bin.
%   .confMat     : Optionally returned field (if 'saveConfMat' = true)
%                  consisitng of an nBins x nCvReps x nClasses x
%                  nClasses array where the (i,j,:,:) slice contains the
%                  nClasses x nClasses confusion matrix (rows: true
%                  class, columns: predicted class) for the j_th repetition
%                  in the i_th bin.
%   .labels      : Optionally returned field (if 'saveLabels' = true and
%                  'oversampleTest' = false) consisting of an nBins x
%                  nCvReps x n_observations numeric array, where the i_th
%                  row contains the predicted labels (aggregated across
%                  folds) for the i_th repetition.
%   .scores      : Optionally returned field (if 'saveScores' = true and
%                  'oversampleTest' = false) consisting of an nBins x
%                  nCvReps x n_observations x nClasses numeric array,
%                  where the (i,j,:,:) slice contains the n_observations x
%                  nClasses matrix for the j_th repetition in the i_th
%                  bin, and the k_th l_th element (of this slice) is the
%                  score for the k_th observation being classified into the
%                  l_th class.
%   .sig         : Scalar struct with the following fields (all individual
%                  null performance values, from whose aggregate the
%                  following measures are calculated, are generated from
%                  only one repetition of a given permutation).
%       .p       : Scalar struct with the following fields (note: these p
%                  values are calculated using an observed performance
%                  values that are the average across repetitions).
%           .accuracy    : nBins x 1 vector of p values for micro-averaged
%                         accuracy.
%           .balaccuracy : nBins x 1 vector of p values for balanced
%                          accuracy.
%           .precision   : nBins x nClasses matrix of p values for precision.
%           .recall      : nBins x nClasses matrix of p values for recall. 
%           .fmeasure    : nBins x nClasses matrix of p values for f-measure.
%           .aucroc      : nBins x nClasses matrix of p values for AUC ROC.
%           .aucpr       : nBins x nClasses matrix of p values for AUC PR.
%       .nullInt : Scalar struct with the following fields (size of
%                  interval dictated by the 'nullInt' name-value pair.
%           .accuracy    : nBins x 2 matrix whose i_th row has as its 1st
%                          and 2nd elements the lower and upper bounds of
%                          the interval on the accuracy null distribution
%                          for the i_th bin.
%           .balaccuracy : nBins x 2 matrix whose i_th row has as its 1st
%                          and 2nd elements the lower and upper bounds of
%                          the interval on the balanced accuracy null
%                          distribution for the i_th bin.
%           .precision   : nBins x nClasses x 2 array where the (i,:,:)
%                          slice is a matrix whose j_th row has as its 1st
%                          and 2nd elements the lower and upper interval
%                          bounds on the precision null distribution, where
%                          the j_th class is positive, in the i_th bin.
%           .recall      : nBins x nClasses x 2 array where the (i,:,:)
%                          slice is a matrix whose j_th row has as its 1st
%                          and 2nd elements the lower and upper interval
%                          bounds on the recall null distribution, where
%                          the j_th class is positive, in the i_th bin.
%           .fmeasure    : nBins x nClasses x 2 array where the (i,:,:)
%                          slice is a matrix whose j_th row has as its 1st
%                          and 2nd elements the lower and upper interval
%                          bounds on the f-measure null distribution, where
%                          the j_th class is positive, in the i_th bin.
%           .aucroc      : nBins x nClasses x 2 array where the (i,:,:)
%                          slice is a matrix whose j_th row has as its 1st
%                          and 2nd elements the lower and upper interval
%                          bounds on the AUC ROC null distribution, where
%                          the j_th class is positive, in the i_th bin.
%           .aucpr       : nBins x nClasses x 2 array where the (i,:,:)
%                          slice is a matrix whose j_th row has as its 1st
%                          and 2nd elements the lower and upper interval
%                          bounds on the AUC PR null distribution, where
%                          the j_th class is positive, in the i_th bin.
%       .zscore  : Scalar struct with the following fields:
%           .accuracy    : nBins x 1 vector whose i_th element is the
%                          number of standard deviations from the mean of
%                          the null accuracy distribution that the observed
%                          accuracy (averaged across repetitions) lies, for
%                          the i_th bin.
%           .balaccuracy : nBins x 1 vector whose i_th element is the
%                          number of standard deviations from the mean of
%                          the null balanced accuracy distribution that the
%                          observed accuracy (averaged across repetitions)
%                          lies, for the i_th bin.
%           .precision   : nBins x nClasses matrix whose i_th j_th
%                          element is the number of standard deviations
%                          from the mean of the null precision distribution
%                          (for the j_th class as positive) that the
%                          observed precision (averaged across repetitions)
%                          lies, for the i_th bin.
%           .recall      : nBins x nClasses matrix whose i_th j_th
%                          element is the number of standard deviations
%                          from the mean of the null recall distribution
%                          (for the j_th class as positive) that the
%                          observed recall (averaged across repetitions)
%                          lies, for the i_th bin.
%           .fmeasure    : nBins x nClasses matrix whose i_th j_th
%                          element is the number of standard deviations
%                          from the mean of the null f-measure distribution
%                          (for the j_th class as positive) that the
%                          observed f-measure (averaged across repetitions)
%                          lies, for the i_th bin.
%           .aucroc      : nBins x nClasses matrix whose i_th j_th
%                          element is the number of standard deviations
%                          from the mean of the null AUC ROC distribution
%                          (for the j_th class as positive) that the
%                          observed AUC ROC (averaged across repetitions)
%                          lies, for the i_th bin.
%           .aucpr       : nBins x nClasses matrix whose i_th j_th
%                          element is the number of standard deviations
%                          from the mean of the null AUC PR distribution
%                          (for the j_th class as positive) that the
%                          observed AUC PR (averaged across repetitions)
%                          lies, for the i_th bin.
% 
% Author: Jonathan Chien

arguments
    T 
    groundTruth 
    nvp.dropInd = []
    nvp.classifier = @fitclinear
    nvp.classCoding = 'onevsall'
    nvp.kFold = 5 
    nvp.nCvReps = 25
    nvp.cvFun = 1
    nvp.useGpu = false
    nvp.oversampleTrain = false
    nvp.oversampleTest = false
    nvp.condLabels = []
    nvp.nResamples = 100  
    nvp.pval = false 
    nvp.nullInt = 95
    nvp.nPerms = 1000
    nvp.permute = 'labels'
    nvp.saveLabels = false
    nvp.saveScores = false
    nvp.saveConfMat = false
    nvp.concatenate = true
end

% Check and modify inputs, if necessary.
[T, groundTruth, inputStatus] = parse_inputs(T, groundTruth);

% Remove any neurons/features specified by user.
T(:,:,nvp.dropInd) = [];
nBins = size(T, 1);


%% Train and test on empirical data

performance = cell(nBins, 1);

for iBin = 1:nBins 
    % Obtain slice S corresponding to nTrials x nNeurons for current bin,
    % as well as matching trial labels.
    S = squeeze(T(iBin,:,:));
    currGroundTruth = groundTruth(:,iBin);
    
    % Train and test current slice using cross-validation; attach metrics
    % of signficance.
    performance{iBin} ...
        = classifier_significance(S, currGroundTruth, ...
                                  'classifier', nvp.classifier, ...
                                  'classCoding', nvp.classCoding, ...
                                  'cvFun', nvp.cvFun, ...
                                  'useGpu', nvp.useGpu, ...
                                  'oversampleTrain', nvp.oversampleTrain, ...
                                  'oversampleTest', nvp.oversampleTest, ...
                                  'condLabels', nvp.condLabels, ...
                                  'nResamples', nvp.nResamples, ...
                                  'kFold', nvp.kFold, ...
                                  'nCvReps', nvp.nCvReps, ...
                                  'nPerms', nvp.nPerms, ...
                                  'permute', nvp.permute, ...
                                  'saveLabels', nvp.saveLabels, ...
                                  'saveLabels', nvp.saveLabels, ...
                                  'saveConfMat', nvp.saveConfMat, ...
                                  'saveScores', nvp.saveScores, ...
                                  'pval', nvp.pval, ...
                                  'nullInt', nvp.nullInt);
end

% Optionally concatenate fields to return a single struct with
% multi-dimensional array fields, rather than a cell array of structs.
if nvp.concatenate, performance = combine_bins(performance); end

end


% --------------------------------------------------
function [T,groundTruth,inputStatus] = parse_inputs(T,groundTruth)
% Check and parse array inputs, expanding as necessary to match.

% Check input shapes.
assert(ismatrix(groundTruth), ['groundTruth must be passed in as either a ' ...
                               'vector or matrix.'])
assert(length(size(T)) == 2 || length(size(T)) == 3, ...
       'T must be passed in as either a 2D or 3D array.')

% Multiple sets of class labels.
if ~any(size(groundTruth) == 1) % If 2D array with no singleton dims
    nBins = size(groundTruth, 2);
    
    % Only one predictor matrix. Make copies to match number of sets of
    % class labels.
    if ismatrix(T)
        T = repmat(T, 1, 1, nBins);
        T = permute(T, [3 1 2]);
        inputStatus = ['T has been mutated, with copies made to match ' ...
                       'the number of sets of class labels. groundTruth ' ...
                       'remains the same.'];
    else
        % If multiple predictor matrices, must match number of sets of
        % class labels.
        assert(size(T, 1) == nBins, ...
               ['If groundTruth is passed in as an nObs x nDatasets matrix ' ...
                'and T is passed in as an nDatasets x nObs x nFeatures ' ...
                'matrix, the 2nd and 1st dimensions of groundTruth and ' ...
                'T, respectively, must match.'])
        inputStatus = 'T and groundTruth remain the same as passed in.';
    end

% Single set of class labels.
elseif any(size(groundTruth) == 1) 
    % If only one predictor matrix, append extra singleton dimension (first
    % dimension of new array) but warn user that other functions may be
    % more direct/preferrable.
    if ismatrix(T)
        warning(['If decoding is desired only for one predictor matrix and ' ...
                 'one set of class labels, consider calling ' ...
                 'classifier_significance (if measures of statistical ' ...
                 'significance are desired), crossval_model_1, or ' ...
                 'crossval_model_2 directly.'])
        T = reshape(T, 1, size(T, 1), size(T, 2));
        inputStatus = ['T has been mutated, with a new singleton axis ' ...
                       'added as the first array dimension. groundTruth' ...
                       'remains the same.'];
    else
        nBins = size(T, 1);
        groundTruth = repmat(groundTruth, 1, nBins);
        inputStatus = ['Copies of groundTruth were made, equal in number ' ...
                       'to the first dimension size of T. T remains ' ...
                       'the same.'];
    end

else
    error('groundTruth must be either a 2D or 3D array.')
end

end


% --------------------------------------------------
function performanceCat = combine_bins(performance)
% As of this writing, MATLAB treates an indexing request into a.x as an
% index into a; as such, even though the parfor iteration loops are clearly
% order-independent, they cannot be proven so to MATLAB. Hence, the
% combination across bins (run in parallel) is messy and is handled
% separately here, though there are surely better solutions than this one.
% This helper function takes in an nBins x 1 cell array, where each cell
% contains the performance output of a call to classifier_significance on
% one dataset (bin). Returns a struct containing the same field names as
% each performance struct, but with results across all bins concatenated in
% a single numeric array.


nBins = length(performance);
performanceCat = struct();

% Get metric field names. metricNames refer to fields storing performance
% metrics e.g. accuracy, fmeasure, etc. Thus, we exlude the field 'sig',
% which contains substructures with significance measures. Do not modify
% the iterable inside the loop.
iRemove = [];
metricNames = fieldnames(performance{1});
for iMetric = 1:length(metricNames) % remove 'sig' field if it exists
    if strcmp(metricNames{iMetric}, 'sig')
        iRemove = iMetric;
    end
end
metricNames(iRemove) = [];

% Get significance field names. sigNames refers to fields storing
% significance info, e.g. p, nullInt, etc (if significance was assessed,
% otherwise sigNames will be empty).
if isfield(performance{1}, 'sig')
    sigNames = fieldnames(performance{1}.sig);
else
    sigNames = [];
end


% Handle all fields storing metric info.
for iMetric = 1:length(metricNames)
    % Get size of all nonsingleton dims, append dim of size nBins.
    currSize = size(performance{1}.(metricNames{iMetric})); 
    performanceCat.(metricNames{iMetric}) = NaN( [nBins currSize] );

    % Concatenate across bins. % Three ':' because confusion matrix is
    % inherently 2D and the array storing confusion matrices across
    % repetitions will thus be 3D. MATLAB ignores extra ':'s. Note that
    % this is where classes go from being in rows to being in columns.
    for iBin = 1:nBins
        performanceCat.(metricNames{iMetric})(iBin,:,:,:) ...
            = performance{iBin}.(metricNames{iMetric});     
    end

    % Squeeze, in case there is only one dimension other than nBins. 
    performanceCat.(metricNames{iMetric}) ...
        = squeeze(performanceCat.(metricNames{iMetric}));
end


% Handle fields for signficance. First, remove confusionMat, labels, and
% scores fields from metricNames, if they exist (i.e., if user wishes to
% retain them for the original data), as their significance was not
% assessed. 
iRemove = [];
for iMetric = 1:length(metricNames)
    if any(strcmp(metricNames{iMetric}, {'confusionMat', 'labels', 'scores'}))
        iRemove = [iRemove; iMetric];
    end
end
metricNames(iRemove) = [];

% Concatenate significance values.
if ~isempty(sigNames)
for iSig = 1:length(sigNames)
for iMetric = 1:length(metricNames)    
    % Get size of all nonsingleton dims, append dim of size nBins.
    currSize = size(performance{1}.sig.(sigNames{iSig}).(metricNames{iMetric})); 
    performanceCat.sig.(sigNames{iSig}).(metricNames{iMetric}) ...
        = NaN( [nBins currSize] );

    % Concatenate across bins. % Two ':' because some significance
    % measures, like an interval on the null distribution have two values
    % per class, hence the extra dimension. MATLAB ignores extra ':'s. Note
    % that this is where classes go from being in rows to being in columns.
    for iBin = 1:nBins
        performanceCat.sig.(sigNames{iSig}).(metricNames{iMetric})(iBin,:,:) ...
            = performance{iBin}.sig.(sigNames{iSig}).(metricNames{iMetric});
    end  

    % Squeeze, in case there is only one dimension other than
    % nBins. 
    performanceCat.sig.(sigNames{iSig}).(metricNames{iMetric}) ...
        = squeeze(performanceCat.sig.(sigNames{iSig}).(metricNames{iMetric}));
end
end
end

end
