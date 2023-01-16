function factorized = calc_fsd(sizes,targetCcgp,nv)
% Calculates "shattering dimensionality" (SD) for a factorized geometry
% based on Bernardi et al "The Geometry of Abstraction in the Hippocampus
% and Prefrontal Cortex," Cell, 2020.
%
% PARAMETERS
% ----------
% sizes      -- 3-vector whose elements specify the number of trials,
%               neurons, and conditions, respectively, to be used in
%               modeling the factorized model. 
% targetCcgp -- Vector whose elements contain the target CCGPs which we
%               will try to achieve by tuning the hypercuboid representing
%               the factorized model. For 8 conditions, this vector may
%               have 1 - 3 elements; for 4 conditions, it may have 1 or 2
%               elements.
% Name-Value Pairs (nv)
%   'disFactorVals'  -- 2D array of values to be tested over when
%                       tuning lengths of the hypercuboid. Each row
%                       corresponds to one dichotomy and to a set of
%                       factors for tuning one side of the hypercuboid (the
%                       direction orthogonal to a separating hyperplane for
%                       that dichotomy). The maximum number of
%                       rows/decodable dichotomies is 2 for the 4 condition
%                       case and 3 for the 8 condition case (for m
%                       conditions as the vertices of an m-dimensional
%                       hyercube, the instrinsic dimensionality of the
%                       hypercube (and thus the dimension of the embedding
%                       vector space if the hypercube is centered at the
%                       origin) is log2(m) = n; in such an n-dimensional
%                       space, there are n translational degrees of
%                       freedom, and linear separation requires us to be
%                       able to move from one class to another in a single
%                       direction; hence n orthogonal linear separations
%                       are possible). Note however, that the number of
%                       models tested is the number of tuning values raised
%                       to the power of the number of dichotomies, so
%                       overly fine-grained searches may come with a very
%                       high computational cost, especially in the 8
%                       condition case.
%   'tuneWith'       -- (string: 'accuracy'| 'auc') Specify which
%                       classification performance metric should be
%                       compared against targetCcgp (this should correspond
%                       to the same performance metric as targetCcgp).
%   'lossFunction'   -- (string: 'sae'|'sse'|'rss'|'mse') Specify
%                       the function used to calculate error between a
%                       tuned hypercuboid's CCGPs and the passed in target
%                       CCGPs.
%   'tuneOnTrials'   -- ('nTrialsPerCond' (default) | scalar | false)
%                       Specify the number of gaussian trials to be added
%                       to each vertex of the hypercube when tuning. If
%                       'nTrialsPerCond', the function will calculate the
%                       number of trials representing each condtiion in the
%                       actual data and use this value. Or, specifiy some
%                       nonnegative integer. Set to false to tune using
%                       classifiers trained on vertices of hypercube alone
%                       (not recommended for nConditions = 4, since
%                       classifiers will then train on two observations and
%                       usually have accuracy of 0 or 1 (and CCGP is the
%                       average over only 4 classifier's performances).
%                       Note well that this is is not related to the number
%                       of trials added to the vertices of the hypercube
%                       (after tuning) when calculating the SD and final,
%                       full CCGP (see 'nFinalTrials').
%   'useTuningModel' -- (1 | 0 (false)), specify whether or not to save
%                       each factorized model explored (if 'tuneOnTrials'
%                       is not false and single trials were added to
%                       vertices during tuning) and to use the exact
%                       optimal model for final SD and CCGP testing. Note
%                       that 'tuneOnTrials' must not be false if this
%                       argument is set to true. Depending on the system
%                       RAM, search space of displacements, number of
%                       neurons/conditions/trials, setting this to true
%                       may potentially exhaust available memory.
%   'trialVar'       -- (numeric vector). Vector of variances
%                       of the isotropic gaussian distribution from which
%                       we make draws to construct a point cloud added to
%                       each vertex (for each scalar variance, all elements
%                       of the main diagonal of the point cloud covariance
%                       matrix are identically this value, with all off
%                       diagonal elements equal to 0). If 'tuneOnTrials' ~=
%                       false, the function will test all variance values
%                       during the tuning step and select the optimal
%                       variance to apply to the point cloud surrounding
%                       vertices of the final, tuned hypercube during
%                       calculation of SD and the final, full CCGP (or use
%                       the optimal model from tuning--see
%                       'useTuningModel'). Note that if 'tuneOnTrials' =
%                       false, the first element of a vector argument for
%                       trialVar will be selected by default, and only this
%                       value will be used during tuning.
%   'nFinalTrials'  -- ('nTrialsPerCond'(default) | scalar). Specify the
%                       number of trials to be added to the vertices of the
%                       final tuned hypercuboid (for calculation of CCGP,
%                       SD etc.), independently of the number of trials
%                       potentially used around each vertex during tuning.
%                       If 'nTrialsPerCond', the function will infer how
%                       many trials per condition are present in the actual
%                       data and use this value; otherwise, specify a
%                       nonnegative integer as the number of trials around
%                       each vertex. If 'useTuningModel' = true, this
%                       argument is ignored.
%   'zscoreFact'    -- (1 (default) | 0), specify whether or not to
%                      z-score each neuron (across all trials) in the
%                      factorized model. Note that neural variances should
%                      already be close to 1 given enough trials.
%   'calcSd'        -- (1 (default) | 0), specify whether or not to
%                      calculate the SD of the final, tuned, factorized
%                      model. If so, parameters for this routine are set
%                      via the 'sdParams' name-value pair (see the
%                      'decoderParams' name-value pair of the calc_sd.m
%                      function for more information). This can be set to
%                      false to suppress the final SD calculation if for
%                      example, we wish to conduct a rough-grained pilot
%                      search, to be followed by a subsequent fine-grained
%                      search around the optimal parameters of the rough
%                      search.
%   'sdParams'       -- Scalar struct. Once the length of the hypercube is
%                       tuned and gaussian noise added, the shattering
%                       dimenisionality of the factorized model is
%                       calculated via the calc_sd function. This
%                       name-value pair passes in decoding parameters to
%                       calc_sd. See calc_sd documentation for more
%                       information). If passed in as empty (as it is by
%                       default), default parameters will be set inside
%                       calc_sd.
%   'calcCcgp'      -- (1 (default) | 0), specify whether or not to
%                      calculate the CCGP of the final, tuned, factorized
%                      model (this is not related to tuning). If so,
%                      parameters for this routine are set via the
%                      'ccgpParams' name-value pair (but see the note about
%                      the classifier field of 'ccgpParams' below). This
%                      can be set to false to suppress the final CCGP
%                      calculation if, for example, we wish to conduct a
%                      rough-grained pilot search, to be followed by a
%                      subsequent fine-grained search around the optimal
%                      parameters of the rough search.
%   'ccgpParams'    -- Scalar struct. Once the length of the hypercuboid is
%                      tuned and gaussian noise added, the CCGP of the
%                      factorized model can be calculated via the calc_ccgp
%                      function. For this routine, specify the name-value
%                      pairs for the calc_ccgp function in the fields of a
%                      scalar struct passed in through 'ccgpParams'. The
%                      passed in values will be checked, and if any
%                      parameters are not specified, default values will be
%                      specified (see local function parse_ccgp_inputs;
%                      note that these default settings will override the
%                      default name-value pair values in the calc_ccgp
%                      function). NB: the value of ccgpParams.classifier
%                      will be used to specify the classifier used both to
%                      calculate the final CCGP and during the tuning step.
%                      Unless otherwise specified, it is set as
%                      @fticlinear; this value is used even if 'calcCcgp'
%                      is false. All other fields of ccgpParams are unused
%                      if 'calcCcgp' is false.
%                               
% RETURNS
% -------
% factorized -- 1 x 1 struct with the following fields:
%   .sd          -- 1 x 1 struct containing shattering dimensionality
%                   information about the tuned factorized model (this is
%                   the output of the calc_sd function when called on the
%                   factorized model).
%   .decoderPerf -- 1 x 1 struct containing information about decoding
%                   performance (accuracy and auc) for each dichotomy of
%                   the factorized model (see decoderPerf under RETURNS in
%                   the documentation for calc_sd for more information).
%                   Currently, there is no option to compute significance
%                   measures for these performances.
%   .ccgp        -- 1 x 1 struct containing CCGP information about the
%                   tuned factorized model (this is the output of the
%                   calc_ccgp function when called on the factorized
%                   model).
%   .fTxN        -- nFinalTrials*nConds x nNeurons matrix whose rows
%                   correspond to single trials in the point clouds around
%                   the hypercuboid vertices.
%   .vertices    -- nConds x nNeurons matrix whose i_th row is the centroid
%                   of the i_th condition (vertex) in the factorized model.
%
% Author: Jonathan Chien 7/24/21. Last edit: 7/8/22.


arguments
    sizes 
    targetCcgp 
    nv.disFactorVals = 1 : 0.1 : 2
    nv.tuneWith = 'accuracy' 
    nv.lossFunction = 'sse'
    nv.useTuningModel = true
    nv.tuneOnTrials = 'nTrialsPerCond'
    nv.trialVar = 1
    nv.nFinalTrials = 'nTrialsPerCond' 
    nv.zscoreFact = true
    nv.sdParams = []
    nv.ccgpParams = []
    nv.calcSd = true
    nv.calcCcgp = true
end


%% Parse and check arguments

% Determine number of trials, neurons, conditions, and trials per
% condition, as well as number of factorized models we will need to tune
% and test (nModels).
nTrials = sizes(1);
nNeurons = sizes(2);
nConds = sizes(3);
assert(mod(nTrials,nConds)==0, 'nTrials must be evenly divisible by nConds.')
nDichotToTune = length(targetCcgp);
nDisFactorVals = length(nv.disFactorVals);
if any(strcmp(nv.trialVar, {'automean', 'automax', 'automin'}))
    nVarVals = 1;
elseif ischar(nv.trialVar)
    error("Invalid string value for 'trialVar'.")
else
    nVarVals = length(nv.trialVar);
end

% Check arguments/options for tuning/trials/variance.
if any(strcmp(nv.trialVar, {'automean', 'automax', 'automin'}))
    assert(~nv.tuneOnTrials, ...
           "If 'trialVar' is set to 'automean', 'automax', or 'automin', " + ...
           "'tuneOnTrials' must be false.")
end
if ~nv.tuneOnTrials & nVarVals > 1
    warning("A range of trial variances was supplied, but tuning was " + ...
            "requested based on vertices only. The first variance value " + ...
            "will be used by default when constructing the final tuned " + ...
            "hypercube, so it is recommended to pass only the desired " + ...
            "point cloud variance as a scalar.")
    nv.trialVar = nv.trialVar(1);
    nVarVals = 1;
end

% Check for potentially problematic values for nConds.
if nConds ~= 8 && nConds ~= 4
    warning('This function currently only supports nConds = 4 or nConds = 8.')
end

% Ensure that the user has not requested the tuning of too many dichotomies.
if nConds == 4
    assert(nDichotToTune <= 2, ...
           ['For a factorized model of 4 conditions, only 2 dichotomies are ' ...
            'decodable.']);
elseif nConds == 8
    assert(nDichotToTune <= 3, ...
           ['For a factorized model of 8 conditions, only 3 dichotomies are ' ...
            'decodable.']);
end

% Expand the tuning factors into an array whose first dim size matches the
% number of dichotomies we would like to tune. Then pad with rows of zeros
% so that the first dim size = 3.
expandedDisFactorVals = repmat(nv.disFactorVals, nDichotToTune, 1);
if nDichotToTune < 3
    expandedDisFactorVals = [expandedDisFactorVals; ...
                             zeros(3-nDichotToTune, nDisFactorVals)];
end


%% Tune hypercuboid

% Parse calc_ccgp parameters (passed in through the 'ccgpParams' name-value
% pair) here, since calc_ccgp is called during the tuning step, and the
% classifier must be set. 
ccgpParamsParsed = parse_ccgp_params(nv.ccgpParams);

% If user opted to add simulated trials cloud around each vertex of the
% hypercube, expand the training labels to match.
if nv.tuneOnTrials
    if strcmp(nv.tuneOnTrials, 'nTrialsPerCond')
        nTuningTrials = nTrials / nConds;
    else
        assert(isscalar(nv.tuneOnTrials) && mod(nv.tuneOnTrials,1)==0, ...
               "If 'tuneOnTrials' is not 'nTrialsPerCond', it must be a " + ...
               "nonnegative integer.")
        nTuningTrials = nv.tuneOnTrials;
    end
    tuningTrialLabels = repelem((1:nConds)', nTuningTrials);
else
    nTuningTrials = 0;
    tuningTrialLabels = (1:nConds)';
end

% Preallocate (same container size regardless of number of conditions due
% to 'omitnan' in min function).
ccgpLoss = NaN(nDisFactorVals, nDisFactorVals, nDisFactorVals, nVarVals);

% Option to save all tuning models and test on the exact model that yielded
% the minimum loss. Note that this can cause RAM to be exceeded depending
% on the size of the search space, number of neurons, number of tuning
% trials, and available RAM.
if nv.useTuningModel
    assert(nv.tuneOnTrials, ...
           "If 'saveTuningModels' is true, 'tuneOnTrials' must be true.")
    tuningModels = NaN(nDisFactorVals, ...
                       nDisFactorVals, ...
                       nDisFactorVals, ...
                       nVarVals, ...
                       nv.tuneOnTrials*nConds, ...
                       nNeurons);
end

% Try all combinations of different values for displacement, which tunes
% the length of the hypercube. 
parfor iFactor1 = 1:nDisFactorVals
    for iFactor2 = 1:nDisFactorVals
        for iFactor3 = 1:nDisFactorVals
            for iVar = 1:nVarVals
            
                % Obtain current set of displacement scaling factors.
                currDisFactors = [expandedDisFactorVals(1,iFactor1) ...
                                  expandedDisFactorVals(2,iFactor2) ...
                                  expandedDisFactorVals(3,iFactor3)];
                if nConds == 4, currDisFactors(3) = []; end
        
                % Using current displacement factors, get vertices of
                % cuboid embedded in N-space, with N = nNeurons. We don't
                % apply sqrt to nvp.trialVar here, because sqrt will be
                % applied in embed_hypercube.
                [hypercuboid, dichotomies] ...
                    = embed_hypercuboid(nNeurons, nConds, currDisFactors, ...
                                        'addTrials', nTuningTrials, ...
                                        'trialVar', nv.trialVar(iVar)); 

                if nv.zscoreFact, hypercuboid = zscore(hypercuboid); end
                
                % Calculate CCGPs of dichotomies for which the current
                % hypercube was tuned. For speed, CCGP is calculated only
                % for these specific dichotomies.
                ccgp = calc_ccgp(hypercuboid, tuningTrialLabels, ...
                                 'classifier', ccgpParamsParsed.classifier, ...
                                 'dichotomies', dichotomies, 'pval', false);

                % Calculate loss using the specified performance metric and
                % loss function.
                ccgpLoss(iFactor1,iFactor2,iFactor3,iVar) ...
                            = calc_loss(targetCcgp, ...
                                        ccgp.(nv.tuneWith).ccgp(1:nDichotToTune), ...
                                        nv.lossFunction);

                % Optionally save current model.
                if nv.useTuningModel
                    tuningModels(iFactor1,iFactor2,iFactor3,iVar,:,:) ...
                        = hypercuboid;
                end
            end
        
            % Breaking condition.
            if nConds == 4 || nDichotToTune < 3, break; end
        end
    
        % Breaking condition.
        if nDichotToTune == 1, break; end
    end
end


%% Construct optimal tuned hypercuboid and calculate CCGP and SD

% Select displacement factors leading to smallest error between target
% CCGPs and constructed CCGP(s). 
[minLoss, iMin] = min(ccgpLoss, [], 'all', 'omitnan', 'linear');
[iFactor1Opt,iFactor2Opt,iFactor3Opt,iVarOpt] ...
    = ind2sub([repmat(nDisFactorVals, 1, 3) nVarVals], iMin);
optDisFactors = [expandedDisFactorVals(1,iFactor1Opt) ...
                 expandedDisFactorVals(2,iFactor2Opt) ...
                 expandedDisFactorVals(3,iFactor3Opt)];
if nConds == 4, optDisFactors(3) = []; end

% If one of the optimal parameters (including trial variance) was the
% largest or smallest provided, warn user that search grid may need to be
% shifted/modified to better cover loss basin.
if any(optDisFactors == nDisFactorVals) 
    warning(['The optimal geometric structure found occured at one of the ' ...
             'upper edges of the supplied search grid. Consider shifting ' ...
             'up the range of the displacmentFactors.'])
elseif iVarOpt == nVarVals && nVarVals ~= 1
    warning(['The optimal geometric structure found occured at the ' ...
             'maximal supplied value for trial variance. Consider shifting ' ...
             'up the range of variances to be tested.'])
elseif any(optDisFactors==1) && ~any(find(optDisFactors==1) > nDichotToTune)
    warning(['The optimal geometric structure found occured at one of the ' ...
             'lower edges of the supplied search grid. Consider shifting ' ...
             'down the range of the displacmentFactors or increasing the ' ...
             'resolution of the search if feasible.'])
elseif iVarOpt == 1 && nVarVals ~= 1
    warning(['The optimal geometric structure found occured at the ' ...
             'minimal supplied value for trial variance. Consider shifting ' ...
             'down the range of variances to be tested or increasing the ' ...
             'resolution of the search, if feasible.'])
end

% Get optimal tuned factorized model.
if nv.useTuningModel
    % If tuning models were saved, use the saved model with minimum loss. 
    Tf = squeeze(tuningModels(iFactor1Opt,iFactor2Opt,iFactor3Opt,iVarOpt,:,:));

    % Set up single trial labels for SD and CCGP.
    calculationTrialLabels = repelem((1:nConds)', nv.tuneOnTrials);
else
    % Else, instantiate hypercube vertices using optimal parameters. First,
    % ensure 'nFinalTrials' has valid value.
    if strcmp(nv.nFinalTrials, 'nTrialsPerCond')
        nFinalTrials = nTrials / nConds; 
    else
        assert(nv.nFinalTrials ~= 0)
        assert(isscalar(nv.nFinalTrials) && mod(nv.nFinalTrials,1)==0, ...
               "If 'nFinalTrials' is not 'nTrialsPerCond', it must be a " + ...
               "nonnegative integer.")
        nFinalTrials = nv.nFinalTrials;
    end

    % Create factorized model.
    Tf = embed_hypercuboid(nNeurons, nConds, optDisFactors, ...
                             'addTrials', nFinalTrials, ...
                             'trialVar', nv.trialVar(iVarOpt));

    if nv.zscoreFact, Tf = zscore(Tf); end

    % Set up single trial labels for SD and CCGP.
    calculationTrialLabels = repelem((1:nConds)', nFinalTrials);
end

clear tuningModels


%% Test factorized model (SD and CCGP)

% Calculate SD for factorizedTxN. Decoding params are passed in through the
% 'sdParams' name-value pair.
if nv.calcSd
    [factorized.sd, factorized.decoderPerf, factorized.sdParams] ...
        = calc_sd(Tf, calculationTrialLabels, 'decoderParams', nv.sdParams);
end 

% Calculate CCGP for factorized model.
if nv.calcCcgp
    factorized.ccgp ...
        = calc_ccgp(Tf, calculationTrialLabels, ...
                    'classifier', ccgpParamsParsed.classifier, ...
                    'pval', ccgpParamsParsed.pval, ...
                    'nullInt', ccgpParamsParsed.nullInt, ...
                    'nNull', ccgpParamsParsed.nNull, ...
                    'nullMethod', ccgpParamsParsed.nullMethod, ...
                    'permute', ccgpParamsParsed.permute, ...
                    'returnNullDist', ccgpParamsParsed.returnNullDist);  
    factorized.ccgpParams = ccgpParamsParsed;
else
    factorized.ccgpParams = ccgpParamsParsed;

    % Remove field names that are not classifier. 
    fnames = fieldnames(factorized.ccgpParams);
    for iField = 1:length(fnames)
        if ~strcmp(fnames{iField}, 'classifier')
            factorized.ccgpParams = rmfield(factorized.ccgpParams, fnames{iField});
        end
    end
end


%% Save results

% Store information related to tuning, including optimal displacement
% factors, loss, and minimum loss.
factorized.optimization.optDisFactors = optDisFactors;
factorized.optimization.ccgpLoss = ccgpLoss;
factorized.optimization.minLoss = minLoss;
factorized.fTxN = Tf;
factorized.vertices = embed_hypercuboid(nNeurons, nConds, optDisFactors, ...
                                        'addTrials', false);


end


% -------------------------------------------------------
function parsedParams = parse_ccgp_params(passedParams)
% Each field in the defaultParams struct is checked against the fields of
% the passed in params (passedParams). If a match is found, the value of
% the field in passedParams is assigned to the field of the same name in
% parsedParams. If no match is found, the value of the field from
% defaultParams is used instead. Note that any extraneous fields in
% passedParams will thus be ignored. 

if isempty(passedParams), passedParams = struct('aaa', []); end

% Create struct with default params.
defaultParams = struct('condNames', [], 'dropInd', [], 'dichotomies', [], ...
                       'classifier', @fitclinear, 'pval', 'two-tailed', ...
                       'nullInt', 95, 'nNull', 1000, ...
                       'nullMethod', 'permutation', 'permute', 'neurons', ...
                       'returnNullDist', false);

% Create a struct with empty fields whose names match those in default
% params.
fnamesDefault = fieldnames(defaultParams);
for iField = 1:length(fnamesDefault)
    parsedParams.(fnamesDefault{iField}) = [];
end

% Get all field names in passed params struct.
fnamesPassed = fieldnames(passedParams);

% For each field in parsedParams, use the passed in value if passedParams
% has a field of the same name; else use the value from defaultParams.
for iField = 1:length(fnamesDefault)
    field = fnamesDefault{iField};

    if cell2mat(strfind(fnamesPassed, field))
        parsedParams.(field) = passedParams.(field);
    else
        parsedParams.(field) = defaultParams.(field);
    end
end

end


% -------------------------------------------------------
function loss = calc_loss(y,yhat,metric)
% Calculate loss between two vector inputs.

assert(isvector(y) && isvector(yhat))
if isrow(y), y = y'; end
if isrow(yhat), yhat = yhat'; end

switch metric
    case 'sae' % sum of absolute erorrs
        loss = sum ( abs ( y - yhat) );
    case 'sse' % sum of squared errors
        loss = sum( (y - yhat).^ 2 );
    case 'rss' % root sum of squares = 2-norm of error vector
        loss = sqrt( sum( (y - yhat).^2 ) );
    case 'mse' % mean squared error
        loss = mean( (y - yhat).^2 ); 
end

end


% -------------------------------------------------------
function [hypercuboid,dichotomies] = embed_hypercuboid(nNeurons,nConds,disFactors,nvp)
% Tune and embed a hypercuboid in n-space, with n = nNeurons. Currently,
% this works only with 4 conditions (in which case the hypercuboid is a
% rectnangle) and with 8 conditions.
% 
% PARAMETERS
% ----------
% nNeurons   -- Scalar value equal to the number of neurons and to the
%               dimensionality of th embedding space.
% nConds     -- Scalar value equal to the number of condition centroids,
%               which are the vertices of the embedded hypercuboid.
% disFactors -- m-vector whose i_th element is a factor multiplying the
%               length of the displacement vector applied to the i_th
%               direction/side of the hypercube. If nConditions = 4, m <=
%               2. If nConditions = 8, m <= 3. If any element of m is zero,
%               the corresponding displacement vector will be 0 (that is,
%               that side will not be scaled).
% Name-Value Pairs (nvp)
%   'addTrials' -- (Scalar|false (default)). Specify the number of trials
%                  (drawn from a Gaussian distribution whose variance is
%                  controlled through 'trialVar' (see below)) to be placed
%                  around each vertex of the hypercuboid. For nConditions =
%                  4, there are only 4 decoders per dichotomy, each trained
%                  on only two points (if only vertices are used) so the
%                  addition of noise allows for smoother exploration of the
%                  loss landscape when tuning CCGP; also, the optimally
%                  tuned variance can be used when simulating trials in the
%                  final calculation of CCGP on the tuned hypercuboid. Set
%                  to false to return only the vertices of the hypercuboid.
%   'trialVar'  -- Scalar value specifying the variance of the Gaussian
%                  distribution from which simulated single trials are
%                  drawn to create point clouds around the condition
%                  centroids. If 'addTrials' is false, this value is
%                  ignored.
%
% RETURNS
% -------
% hypercuboid -- If 'addTrials' is a scalar, this is an
%                nConditions*nTrialsPerCondition x nNeurons array (where
%                nTrialsPerCondition = 'addTrials'); essentially this
%                simulates the neural population representation as 8
%                condition means situated at the vertices of a hypercuboid
%                in n-dimensional space (n = nNeurons), where each vertex
%                is surrounded by a point cloud simulating single trials of
%                that condition. If 'addTrials' = false, this is an
%                nConditions x nNeurons array, simulating the neural
%                population representation as 8 condition means only (i.e.,
%                without the simulated single trials).
% dichotomies -- m x nConditions array, where the i_th row contains
%                condition indices for the dichtomy whose separating
%                hyperplane is orthogonal to the side of the hypercuboid
%                tuned using the i_th element of disFactors. By returning
%                these indices, we can test only the dichotomies
%                corresponding to the scaled sides of the hypercuboid, thus
%                speeding up the grid search. Save for the variable number
%                of rows, this output is exactly the same as the first
%                output of create_dichotomies.
%
% Jonathan Chien. 1/20/22.

arguments
    nNeurons
    nConds
    disFactors
    nvp.addTrials = false % number of trials around each centroid
    nvp.trialVar = 1 % Variance of each point cloud, if trials added vertices
end

% Check for valid displacement factors. Namely, should not be less than
% -0.5, or conditions will be inverted.
if any(disFactors < -0.5)
    error("At least one of the displacement factors passed in was less " + ...
          "than -0.5. This will cause the corresponding side(s) of the " + ...
          "hypercube to become inverted, with condition centroids on " + ...
          "opposite sides of the dichotomy/ies switching places relative " + ...
          "to each other.")
end

% Get number of displacement factors provided (the number of sides whose
% lengths we wish to tune).
nDichotToTune = length(disFactors);

if nConds == 4
    % With n = nNeurons, begin with one random standard normal n-vector,
    % then perform the QR factorization of this vector to obtain Q. Take
    % any two columns of Q and their respective reflections to obtain
    % vertices of a square (each side has length sqrt(2)). Next calculate
    % the vector difference between two orthogonal vectors out of these
    % four vectors to obtain a vector, d1, parallel to two sides of the
    % square; do the same with the other two vertices to get d2. Scale d1
    % and d2 by their respective factors (in the elements of
    % displacementFactor) and apply d1 and d2 to two respective sides of
    % the square.

    % Check number of displacement factors provided.
    assert(nDichotToTune <= 2)
    if nDichotToTune == 1, disFactors = [disFactors 0]; end

    % Set up hypercube (square).
    [Q,~] = qr(randn(nNeurons, 1));
    vertices(1,:) = Q(:,1);
    vertices(2,:) = Q(:,2);
    vertices(3,:) = -vertices(2,:);
    vertices(4,:) = -vertices(1,:);

    % Set up indices of two dichotomies that are tunable.
    dichotomies = [1:4; ...
                   [1 3 2 4]];

    % Calculate displacement vectors, multiply by displacement factors and
    % apply to vertices of hypercube. 
    d = NaN(2, nNeurons);
    d(1,:) = (vertices(1,:) - vertices(3,:)) * disFactors(1); 
    d(2,:) = (vertices(1,:) - vertices(2,:)) * disFactors(2); 
    for iDichot = 1:2
        vertices(dichotomies(iDichot,1:2),:) ...
            = vertices(dichotomies(iDichot,1:2),:) + d(iDichot,:);
        vertices(dichotomies(iDichot,3:4),:) ...
            = vertices(dichotomies(iDichot,3:4),:) - d(iDichot,:);
    end

    % Remove rows of "dichotomies" so that its 1st dim size is equal to
    % nDichotToTune. This can then be passed in to calc_ccgp through the
    % appropriate name-value pair.
    dichotomies = dichotomies(1:nDichotToTune,:);
    
elseif nConds == 8
    % With n = nNeurons, begin with one random standard normal n-vector,
    % and perform QR decomposition. Select two columns of Q and their
    % reflections to define a square. Then use any other column of Q (i.e.,
    % not among the two already selected) scaled by (sqrt(2)/2) as an
    % initial displacement vector, di, to displace the square twice, in
    % opposite directions (each orthogonal to the vector subspace in which
    % the square resides). The first displacment is by adding di, and the
    % second is by adding -di. Next calculate d1, d2, and d3 in a manner
    % analagous to the 4 condition case above and apply them to the
    % vertices of the hypercube, in order to tune the length of its sides.

    % Check number of displacement factors provided.
    assert(nDichotToTune <= 3)
    if nDichotToTune < 3
        disFactors = [disFactors zeros(1, 3-nDichotToTune)];
    end

    % Set up hypercube (3D). Displace vertices 5-8 first.
    [Q,~] = qr(randn(nNeurons, 1));
    vertices(1,:) = Q(:,1);
    vertices(2,:) = Q(:,2);
    vertices(3,:) = -vertices(2,:);
    vertices(4,:) = -vertices(1,:);
    vertices(5:8,:) = vertices(1:4,:) - (sqrt(2)/2) * Q(:,3)';
    vertices(1:4,:) = vertices(1:4,:) + (sqrt(2)/2) * Q(:,3)'; 
    
    % Set up indices of the 3 dichotomies that could be tuned. 
    dichotomies = [1:8; ...
                   [1 2 5 6 3 4 7 8]; ...
                   [1 3 5 7 2 4 6 8]];

    % Calculate displacement vectors (d1, d2, d3), scale them by their
    % respective factors, and apply them to vertices of hypercube.
    d = NaN(3, nNeurons);
    d(1,:) = (vertices(1,:) - vertices(5,:)) * disFactors(1);
    d(2,:) = (vertices(1,:) - vertices(3,:)) * disFactors(2);
    d(3,:) = (vertices(1,:) - vertices(2,:)) * disFactors(3);
    for iDichot = 1:3
        vertices(dichotomies(iDichot,1:4),:) ...
            = vertices(dichotomies(iDichot,1:4),:) + d(iDichot,:);
        vertices(dichotomies(iDichot,5:8),:) ...
            = vertices(dichotomies(iDichot,5:8),:) - d(iDichot,:);
    end

    % Remove rows of "dichotomies" so that its 1st dim size is equal to
    % nDichotToTune. This can then be passed in to calc_ccgp through the
    % appropriate name-value pair.
    dichotomies = dichotomies(1:nDichotToTune,:);

else
    error(['nConds was passed in as %d, which is not currently an ' ...
           'acceptable value.'], nConds)
end

% Option to simulate point cloud of trials around each vertex with draws
% from a normal distribution (variance = 'trialVar').
if nvp.addTrials
    hypercuboid = NaN(nConds * nvp.addTrials, nNeurons);
    for iCond = 1:nConds
        hypercuboid((iCond-1)*nvp.addTrials+1 : iCond*nvp.addTrials, :) ...
            = vertices(iCond,:) ...
              + randn(nvp.addTrials, nNeurons) * sqrt(nvp.trialVar);
    end
else
    hypercuboid = vertices;
end

end
