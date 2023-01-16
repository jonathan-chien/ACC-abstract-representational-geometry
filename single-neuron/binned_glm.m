function regression = binned_glm(neurons, modelNumber, nv)
% Accepts as input neurons, a 1 x nNeurons cell arrays and a modelNumber
% specifying the design matrix. Returns a 1 x 1 struct with various
% regression outputs in the fields.
%
% PARAMETERS
% ----------
% neurons  -- 1 x nNeurons cell array. Each cell contains a 1 x 1 struct 
%             with the following fields:
%   .firingRates  -- nTrials x nBins firing rate matrix for the
%                    current neuron, timelocked to the epoch defining
%                    event. 
%   .trialCodes   -- 1 x nTrials cell array, where the t_th cell contains a
%                    column vector of code numbers that appeared during the
%                    t_th trial. 
%   .pictures     -- nTrials x 1 vector of picture/cue labels (1-4), where
%                    t_th component is the label of the picture that
%                    appeared on the t_th trial.
%   .barSize      -- nTrials x 1 vector of bar sizes, where the t_th
%                    component is the bar size (from 1 to 7) at the
%                    beginning of the t_th trial. (Note that this
%                    convention has 0 bar = 1)
%   .currentTrial -- nTrials x 1 vector of trial indices within current
%                    block. Trials occur in blocks of six, and the t_th
%                    component of this field is the index of the t_th
%                    trial within the current block.
%   .area         -- String value identifying the brain region.
%   .epoch        -- The current epoch. Not used in the function here but
%                    is passed on to the output struct, regression, for
%                    future reference further down the pipeline.
% 'modelNumber'   -- Scalar value (1, 2, or 3) indicating the type of
%                    design matrix to be used. Note that the design
%                    matrices under all three models have five
%                    predictors, the last four of which are the same
%                    across models. The difference lies in the first
%                    predictor which is picture valence under model 1,
%                    absolute feedback valence under model 2, and
%                    relative feedback valence under model 3. 
% Name-Value Pairs (nv)
%   'zscorePreds'    -- Logical true or false specifying whether or not to
%                       normalize the columns of the design matrix to have
%                       mean equal to 0 and variance equal to unity.
%                       Default false.
%   'distribution'   -- String value specifying distribution of response
%                       variable for the GLM. Default is 'normal', which
%                       will cause the link function (under the default
%                       settings of glmfit.m) to be the identity, with f(u)
%                       = u.
%
% RETURNS
% -------
% regression -- 1 x 1 struct with the following fields:
%   .betaVal     -- nNeurons x nBins matrix, where the i_th row
%                   j_th column component is the beta value for the valence
%                   predictor in the model trained on a response vector
%                   whose t_th component is the mean FR of the i_th neuron
%                   during the j_th bin on the t_th trial. Note that this
%                   "valence" can correspond to picture valence, absolute
%                   feedback valence, or relative feedback valence,
%                   depending on whether the specified model number is 1,
%                   2, or 3, respectively.
%   .betaDir     -- The same as .betaVal but the predictor is trial
%                   response direction.
%   .betaInt     -- The same as .betaVal but the predictor is the
%                   interaction between valence and response direction.
%   .betaBar     -- The same as .betaVal but the predictor is the size of
%                   the bar on the given trial (observation in the
%                   regression model).
%   .betaTrial   -- The same as .betaVal but the predictor is the current
%                   trial index within the current trial block (blocks
%                   consist of 6 trials), equivalent to played trials + 1.
%   .pVal        -- nNeurons x nBins matrix, where the i_th row
%                   j_th column component is the beta value for the valence
%                   predictor in the model trained on a response vector
%                   whose t_th component is the mean FR of the i_th neuron
%                   during the j_th bin on the t_th trial. (See note about
%                   "valence" under .betaVal above).
%   .pDir        -- The same as .pVal but the predictor is trial response 
%                   direction.
%   .pInt        -- The same as .pVal but the predictor is the interaction
%                   between valence and trial response direction.
%   .pBar        -- The same as .pVal but the predictor is the size of the
%                   bar on the given trial (observation in the regression
%                   model).
%   .pTrial      -- The same as .pVal but the predictor is the current
%                   trial index within the current trial block (blocks
%                   consist of six trials), equivalent to played trials +
%                   1.
%   .cpdVal      -- nNeurons x nBins matrix, where the i_th row
%                   j_th column component is the coefficient of partial
%                   determination for the valence predictor in the model
%                   trained on a response vector whose t_th component is
%                   the mean FR of the i_th neuron during the j_th bin on
%                   the t_th trial. (See note about "valence" under
%                   .betaVal above).
%   .cpdDir      -- The same as .cpdVal but the predictor is trial response
%                   direction.
%   .cpdInt      -- The same as .cpdVal but the predictor is the
%                   interaction between valence and response direction.
%   .cpdBar      -- The same as .cpdVal but the predictor is the size of
%                   the bar on the given trial (observation in the
%                   regression model).
%   .cpdTrial    -- The same as .cpdVal but the predictor is the current
%                   trial index within the current trial block (blocks
%                   consist of 6 trials), equivalent to played trials + 1.
%   .rSqr        -- nNeurons x nBins matrix, where the i_th row
%                   j_th column component is the coefficient of
%                   determination for the model trained on a response
%                   vector whose t_th component is the mean FR of the i_th
%                   neuron during the j_th bin on the t_th trial.
%   .vif         -- nNeurons x nPredictors matrix, where the i_th
%                   row j_th column component is the variance inflation
%                   factor (VIF) for the j_th predictor in all models of
%                   the i_th neuron (across all b bins, where each model is
%                   trained on a response vector whose t_th element is the
%                   mean FR of the i_th neuron on the t_th trial in the
%                   b_th bin). Note that nPredictors = nCols of the design
%                   matrix, without counting the intercept column, and that
%                   the cols of the .vif field match the order of the cols
%                   of the designMatrix (again minus the intercept column).
%                   Note as well that many of the rows in .vif will be
%                   identical, as neurons (rows) from the same session see
%                   the same trials and thus share the same design matrices
%                   and VIF values (which are calculated solely from the
%                   design matrix, which is formed in turn from the
%                   trials). The number of unique rows in .vif will thus be
%                   equal to nSessions.
%   .areas       -- nNeurons x 1 where the i_th component is the area code
%                   for the i_th neuron.
%   .modelNumber -- The model used for regression (can be 1, 2, or 3).
%   .epoch       -- String value denoting epoch over which regression was
%                   run (derived from the epoch field of the struct in each
%                   cell of the input cell array neurons (the first cell
%                   is arbitrarily chosen)).
%
% Author: Jonathan Chien. 


arguments
    neurons
    modelNumber 
    nv.zscorePreds = false
    nv.distribution string = 'normal'
    nv.link string = 'identity'
end


% Get/check number of neurons and bins (must have same number of bins for
% all neurons).
nNeurons = length(neurons);
nBins = nan(nNeurons, 1);
for i_neuron = 1:nNeurons
    nBins(i_neuron) = size(neurons{i_neuron}.firingRates, 2);
end
assert(range(nBins) == 0, 'All neurons must have the same number of bins.')
nBins = nBins(1);

                                     
%% Fit OLS model for each neuron.

% Preallocate struct regression for subsequent loop, initialize waitbar.
regression = struct('betaVal', NaN(nNeurons, nBins), ...
                    'betaDir', NaN(nNeurons, nBins), ...
                    'betaInt', NaN(nNeurons, nBins), ...
                    'betaBar', NaN(nNeurons, nBins), ...
                    'betaTrial', NaN(nNeurons, nBins), ...
                    'pVal', NaN(nNeurons, nBins), ...
                    'pDir', NaN(nNeurons, nBins), ...
                    'pInt', NaN(nNeurons, nBins), ...
                    'pBar', NaN(nNeurons, nBins), ...
                    'pTrial', NaN(nNeurons, nBins), ...
                    'cpdVal', NaN(nNeurons, nBins), ...
                    'cpdDir', NaN(nNeurons, nBins), ...
                    'cpdInt', NaN(nNeurons, nBins), ...
                    'cpdBar', NaN(nNeurons, nBins), ...
                    'cpdTrial', NaN(nNeurons, nBins), ...
                    'rSqr', NaN(nNeurons, nBins), ...
                    'rmse', NaN(nNeurons, nBins), ...
                    'dev', NaN(nNeurons, nBins), ...
                    'aic', NaN(nNeurons, nBins), ...
                    'vif', NaN(nNeurons, 5), ... % 5 predictors
                    'areas', NaN(nNeurons, 1));
w = waitbar(0, '');
                   
% Iterate over neurons and for each neuron, fit a regression model over
% each time bin.
for iNeuron = 1:nNeurons   

    % Update waitbar.
    waitbar(iNeuron / nNeurons, w, ...
            sprintf('Processing neuron %d of %d...', ...
                    iNeuron, nNeurons));
    
    % Create design matrix (shared across all bins). This will also be the
    % same across all single neurons from the same session.
    switch modelNumber
        case {1, 3}
            designMat ...
                = construct_design_matrix(neurons{iNeuron}, ...
                                          modelNumber, ...
                                          'zscore', nv.zscorePreds);
        case 2
            [designMat, removeInd] ...
                = construct_design_matrix(neurons{iNeuron}, ...                                         
                                          modelNumber, ...
                                          'zscore', nv.zscorePreds);
            neurons{iTrial}.firingRates(removeInd,:) = [];
            neurons{iTrial}.trialCodes(removeInd) = [];
            neurons{iTrial}.pictures(removeInd) = [];
            neurons{iTrial}.barSize(removeInd) = [];
            neurons{iTrial}.currentTrial(removeInd) = [];  
    end
    
    % Calculate VIF (variance inflation factor) for design matrix.
    regression.vif(iNeuron,:) = diag(inv(corrcoef(designMat)));
    
    % Get current neuron's firingRates matrix (trials x timepoints). Option to
    % smooth along 2nd (temporal) dimension (not advised, see
    % documentation).
    firingRates = neurons{iNeuron}.firingRates;
    
    % For each bin, fit a model for current neuron.
    for iBin = 1:nBins        
        % Get nTrials x binWidth for current bin and collapse temporal
        % (2nd) dimension by averaging along it.
        currentBin = firingRates(:,iBin);
        
        % Fit GLM.
        [~,dev,stats] = glmfit(designMat, currentBin, nv.distribution, 'Link', nv.link);
        
        % Calculate RMSE.
        rmse = sqrt(mean((stats.resid).^2));
        
        % Calculate coefficients of determination and partial
        % determination.
        [cd,~,cpd] = calc_cd_cpd(designMat, currentBin, nv.distribution, nv.link);
        
        % Store output in struct to be returned by function. First element
        % of stats.beta and stats.p corresponds to the coefficient term.
        regression.betaVal(iNeuron,iBin) = stats.beta(2);  
        regression.betaDir(iNeuron,iBin) = stats.beta(3);
        regression.betaInt(iNeuron,iBin) = stats.beta(4);
        regression.betaBar(iNeuron,iBin) = stats.beta(5);
        regression.betaTrial(iNeuron,iBin) = stats.beta(6);
        regression.pVal(iNeuron,iBin) = stats.p(2);
        regression.pDir(iNeuron,iBin) = stats.p(3);
        regression.pInt(iNeuron,iBin) = stats.p(4);
        regression.pBar(iNeuron,iBin) = stats.p(5);
        regression.pTrial(iNeuron,iBin) = stats.p(6);
        regression.cpdVal(iNeuron,iBin) = cpd(1);
        regression.cpdDir(iNeuron,iBin) = cpd(2);
        regression.cpdInt(iNeuron,iBin) = cpd(3);
        regression.cpdBar(iNeuron,iBin) = cpd(4);
        regression.cpdTrial(iNeuron,iBin) = cpd(5);
        regression.rSqr(iNeuron,iBin) = cd;
        regression.rmse(iNeuron,iBin) = rmse;
        regression.dev(iNeuron,iBin) = dev;
        regression.areas(iNeuron) = neurons{iNeuron}.area;
    end

end
close(w)

% Additional useful housekeeping information.
regression.modelNumber = modelNumber;
regression.epoch = neurons{1}.epoch;

end


% --------------------------------------------------
function [designMatrix,varargout] = construct_design_matrix(singleNeuron,modelNumber,nv)
% [designMatrix] = construct_design_matrix(singleNeuron, [Name-Value Pairs])
% --------------------------------------------------------------------------
% Takes as input one cell of the allNeurons cell array (see binned_glm) and
% creates a design matrix for single-neuron regression based on a few
% preset template options.
%
% [designMatrix, removeInd] = construct_design_matrix(singleNeuron, [Name-Value Pairs])
% -------------------------------------------------------------------------------------
% Alternative syntax also returns a vector of indices of trials removed in
% order to minimize the number of trials where absolute feedback is
% correlated with picture type (trial types 1 and 4), while keeping numbers
% of each type relatively balanced.
%
% PARAMETERS
% ----------
% singleNeuron -- One cell, e.g. from the nNeuronsROI x 1 cell array in
%                 binned_glm. Briefly, this is expected to be a 1 x 1
%                 struct with the following fields (note that only the
%                 pictures, trialCodes, barSize, and currentTrial fields
%                 are used in this function):
%   .firingRates  -- nTrials x nBins firing rate matrix for the
%                    current neuron, timelocked to the epoch defining
%                    event. 
%   .trialCodes   -- 1 x nTrials cell array, where the t_th cell contains a
%                    column vector of code numbers that appeared during the
%                    t_th trial. 
%   .pictures     -- nTrials x 1 vector of picture/cue labels (1-4), where
%                    t_th component is the label of the picture that
%                    appeared on the t_th trial.
%   .barSize      -- nTrials x 1 vector of bar sizes, where the t_th
%                    component is the bar size (from 1 to 7) at the
%                    beginning of the t_th trial. (Note that this
%                    convention has 0 bar = 1)
%   .currentTrial -- nTrials x 1 vector of trial indices within current
%                    block. Trials occur in blocks of six, and the t_th
%                    component of this field is the index of the t_th
%                    trial within the current block.
%   .area         -- String value identifying the brain region.
%   .epoch        -- The current epoch. Not used in the function here but
%                    is passed on to the output struct, regression, for
%                    future reference further down the pipeline.
% 'modelNumber'   -- Scalar value (1, 2, or 3) indicating the type of
%                    design matrix to be used. Note that the design
%                    matrices under all three models have five
%                    predictors, the last four of which are the same
%                    across models. The difference lies in the first
%                    predictor which is picture valence under model 1,
%                    absolute feedback valence under model 2, and
%                    relative feedback valence under model 3. 
% Name-Value Pairs (nv)
%   'zscore'      -- Logical true or false. If set true, columns of design
%                    matrix, corresponding to predictors, will be z-scored.
%                    Default false.
%
% RETURNS
% -------
% designMatrix -- nObs x nPredictors matrix against which the
%                 trial-by-trial vector of firing rates for a single unit
%                 will be regressed. 
% removeInd    -- An optional argument that is a vector of trial indices
%                 marked for removal in the event that the specified model
%                 number is 2 (see alternative syntax above). This variable
%                 is returned into the workspace of the invoking function
%                 so that the marked trials can also be dropped from the
%                 fields of singleNeuron in that function's workspace.
%                 Perhaps slightly less flexible would be to drop those
%                 trials here and return a modified/trimmed version of
%                 singleNeuron.
% 
% Author: Jonathan Chien Version 1.1. 6/24/21. Last edit: 6/27/21.


arguments
    singleNeuron
    modelNumber 
    nv.zscore = false
end

% Determine number of trials.
nTrials = length(singleNeuron.pictures);

% Fill nTrials x 9 predictors template.
designMatrix = NaN(nTrials, 9);
picType = NaN(nTrials, 1);
for iTrial = 1:nTrials
    
    % Picture valence.
    if singleNeuron.pictures(iTrial) == 1 ...
       || singleNeuron.pictures(iTrial) == 2
        designMatrix(iTrial, 1) = 1;
    elseif singleNeuron.pictures(iTrial) == 3 ...
       || singleNeuron.pictures(iTrial) == 4
        designMatrix(iTrial, 1) = -1;
    end
    
    % Absolute feedback valence.
    if ismember(16, singleNeuron.trialCodes{iTrial})
        designMatrix(iTrial, 2) = 1;
    elseif ismember(15, singleNeuron.trialCodes{iTrial})
        designMatrix(iTrial, 2) = 0;
    elseif ismember(14, singleNeuron.trialCodes{iTrial})
        designMatrix(iTrial, 2) = -1;
    end
    
    % Relative valence. Also calculate minimum number of each picture type; 
    % later this will be used to maximize the proportion of trials
    % featuring 0 bar change (codes.pictures 2 and 3), as this outcome is not
    % correlated with picture valence the way outcomes of +1 and -1 are.
    if (singleNeuron.pictures(iTrial) == 1 || singleNeuron.pictures(iTrial) == 2)... 
       && ismember(16, singleNeuron.trialCodes{iTrial})
        designMatrix(iTrial,3) = 1;
        picType(iTrial) = 1; % pos pic, bar gain
    elseif (singleNeuron.pictures(iTrial) == 1 || singleNeuron.pictures(iTrial) == 2)...
           && ismember(15, singleNeuron.trialCodes{iTrial})
        designMatrix(iTrial,3) = -1;
        picType(iTrial) = 2; % pos pic, bar same
    elseif (singleNeuron.pictures(iTrial) == 3 || singleNeuron.pictures(iTrial) == 4)...
           && ismember(15, singleNeuron.trialCodes{iTrial})
        designMatrix(iTrial,3) = 1;
        picType(iTrial) = 3; % neg pic, bar same
    elseif (singleNeuron.pictures(iTrial) == 3 || singleNeuron.pictures(iTrial) == 4)...
           && ismember(14, singleNeuron.trialCodes{iTrial})
        designMatrix(iTrial,3) = -1;
        picType(iTrial) = 4; % neg pic, bar loss   
    end
    
    % response direction
    if ismember(23, singleNeuron.trialCodes{iTrial}) 
        designMatrix(iTrial, 4) = 1; % left
    elseif ismember(24, singleNeuron.trialCodes{iTrial})
        designMatrix(iTrial, 4) = -1; % right
    end
    
    % pic val x dir interaction effect
    designMatrix(iTrial,5) = designMatrix(iTrial,1)*designMatrix(iTrial,4);
   
    % abs feedback val x dir interaction effect
    designMatrix(iTrial,6) = designMatrix(iTrial,2)*designMatrix(iTrial,4);
   
    % rel feedback val x dir interaction effect
    designMatrix(iTrial,7) = designMatrix(iTrial,3)*designMatrix(iTrial,4);
   
    % ln-transformed bar size
    designMatrix(iTrial,8) = log(singleNeuron.barSize(iTrial));
   
    % ln-transformed trials played
    designMatrix(iTrial,9) = log(singleNeuron.currentTrial(iTrial));
end

% Prune template according to specified model number to create desired
% design matrix.
switch modelNumber
    case 1 % Pic valence + other predictors
        designMatrix = designMatrix(:, [1 4 5 8 9]);
        assert(nargout == 1)
    case 2 % Abs Fdback valence + other predictors
        designMatrix = designMatrix(:, [2 4 6 8 9]);
        if nargout ~= 2
            warning(['removeInd, containing indices of trials removed ' ...
                     'was not requested as an output.'])
        end
        
        % Find trial type with least number of trials.
        nPicType = NaN(4,1);
        for iType = 1:4
            nPicType(iType) = sum(picType == iType);
        end
        minTrial = min(nPicType);
        
        % For each trial type, calculate difference between trial number
        % and minTrial; randomly mark for removal a number of trials
        % equivalent to this difference.
        removeInd = [];
        for iType = 1:4
            removeInd = [removeInd; ...
                         datasample(find(picType == iType),...
                                    nPicType(iType)-minTrial, 'Replace', false)];
        end
        designMatrix(removeInd, :) = [];
        varargout{1} = removeInd;
        
    case 3 % Rel fdback valence + other predictors
        designMatrix = designMatrix(:, [3 4 7 8 9]);
        assert(nargout == 1);
end

% Center ln-transformed predictors after removing incorrect/incomplete
% trials; note that all 3 current models have 5 predictor variables each,
% with ln(barSize) and ln(trialsPlayed) as the fourth and fifth predictors,
% respectively, so it is currently sufficient to operate on any of the
% switch-case outcomes by indexing to the fourth and fifth columns of the
% resulting predictors matrix. 
meanLnBarSize = mean(designMatrix(:,4));
meanLnCurrentTrial = mean(designMatrix(:,5));
designMatrix(:,4) = designMatrix(:,4) - meanLnBarSize;
designMatrix(:,5) = designMatrix(:,5) - meanLnCurrentTrial;

% Last check for any NaNs.
if any(isnan(designMatrix), 'all') 
    warning('NaNs present in design matrix.')
end

% Option to z-score predictors.
if nv.zscore
    designMatrix = normalize(designMatrix);
end

end


% --------------------------------------------------
function [cd,cdAdj,cpd] = calc_cd_cpd(predictors,respVariable,distr,link) 
% Calculates and returns the coefficients of determination and partial
% determination. Also calculates coefficient of determination adjusted for
% number of predictors.
%
% PARAMETERS
% ----------
% predictors   -- m x n predictor matrix, where m is the number of trials and
%                 n the number of predictors.
% respVariable -- m x 1 vector of response variable values, where m is the
%                 number of trials.
% nPredictors  -- Number of predictors. REMOVED as argument on 7/13/21, as
%                 this value is easily derived from the predictors
%                 argument.
% distr        -- Distribution of response variable. Default = 'normal'.
% link         -- Link function. Default = 'identity'.
%
% OUTPUTS
% -------
% cd    -- Coefficient of determination.
% cdAdj -- Coefficient of determination, adjusted for number of predictors
%           (adjusted R^2).
% cpd   -- Coefficient of partial determination.
%
% Adapted from CPD_elr by Erin L. Rich. 


assert(size(predictors, 1) == size(respVariable, 1), ...
       'Dimensions of independent and dependent variables not consistent.')

if sum(isnan(predictors), 'all') > 0
    warning('NaNs present in predictor matrix.')
end

if sum(isnan(respVariable)) > 0
    warning('NaNs present in response variable vector.')
end

% Determine number of predictors and observations.
nPredictors = size(predictors, 2);
nObs = size(predictors, 1);

[~,~,stats] = glmfit(predictors, respVariable, distr, 'Link', link);
SSres = sum(stats.resid.^2); % calculates sum of squared residuals

% Calculate R^2 and adjusted R^2. Note that for the latter, the number of
% predictors includes the intercept (and -(p+1) = -p-1).
meanY = mean(respVariable);
totVar = respVariable - meanY;
SStot = sum(totVar.^2); % calculate total sum of squares
cd = 1 - (SSres/SStot); % cd = coefficient of determination
cdAdj = 1 - ((nObs-1)/(nObs-nPredictors-1)) * (SSres/SStot);

% Calculate CPDs.
cpd = NaN(1, nPredictors); 
for iPredictor = 1:nPredictors
    predictors_i = predictors;
    predictors_i(:, iPredictor) = [];  %remove in turn each column from predictors
    [~,~,stats] = glmfit(predictors_i, respVariable, distr, 'Link', link);
    SSres_i = sum(stats.resid.^2); %sum of squared residuals, reduced
    cpd(iPredictor) = (SSres_i-SSres)/SSres_i; % added the 1 here in first index since matrix was preallocated
end

end
