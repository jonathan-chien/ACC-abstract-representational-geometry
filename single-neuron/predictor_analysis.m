function analysis = predictor_analysis(regression,predictor,nvp)
% Takes as input the output of one call to binned_glm, as well as a string
% specifying a predictor or interest, and returns a series of
% statistics/values based on an analysis of the specified predictor within
% the given regression run.
%
% PARAMETERS
% ----------
% regression -- Output of binned_glm for one run. Briefly, a 1 x 1 struct
%               with the results of regression in its respective fields.
%               See binned_glm documentionation under RETURNS for a
%               complete list.
% predictor  -- String value specifying the predictor of interest for which
%               we would like to analyze the regression results. Currently
%               accepted values are 'valence' and 'direction'. This value
%               may be referred to throughout this documentation as the
%               "specified predictor."
% Name-Value Pairs (nvp)
%   'epochBounds'          -- 2-vector specifying the first and last
%                             timepoint, in ms, of the entire epoch (the
%                             period of time over which all bins from the
%                             regression run were drawn). Timepoints should
%                             be given with respect to the epoch-defining
%                             event (e.g. [-999 1500], with stimulus onset
%                             at 0). Passed to get_bin_timepoints.
%   'analysisWindowBounds' -- 2-vector specifying the first and last
%                             timepoint of the analysis window, a temporal
%                             subset of the entire epoch (wth the epoch
%                             being the period of time over which all bins
%                             from the regression run were drawn). As with
%                             'epochBounds', timepoints should be given
%                             with respect to the epoch-defining event
%                             (e.g., [101 600], with stimulus onset at 0).
%                             Passed to get_bin_timepoints.
%   'binWidth'             -- Scalar value that is the width in ms of a
%                             single bin during regression. Passed to
%                             get_bin_timepoints.
%   'sliding'              -- Logical true or false, specifying whether the
%                             bins in the regression run producing
%                             "regression" were slid with overlap (true) or
%                             not (false). Passed to get_bin_timepoints.
%   'step'                 -- Scalar value that is the amount by which bins
%                             are advanced/slid if 'sliding' = true. Passed
%                             to get_bin_timepoints.
%   'threshold'            -- P value threshold used to assign neurons as
%                             significant (if below threshold) or not (if
%                             above threshold) in conjunction with the
%                             'consecutive' metric.
%   'consecutive'          -- Number of consecutive bins in which a neuron
%                             must have a p value for 'predictor' below the
%                             threshold in order to be deemed signficant.
%   'checkSigIn'           -- String value, either 'analysisWindow' or
%                             'epoch'. If 'analysisWindow', neurons will be
%                             assigned as signficant or not (using criteria
%                             specified through the 'threshold' and
%                             'consecutive' name-value pairs) based on
%                             activity only with the analysis window
%                             specified through 'analysisWindowBounds'. If
%                             'epoch', neurons will be assessed for
%                             signficance using the same criteria but over
%                             the entire epoch specified through
%                             'epochBounds'.
%
% RETURNS
% -------
% analysis -- 1 x 1 struct with the following fields:
%   .sigNxB              -- The same matrix as analysis.unfilteredSigNxB
%                           below, but after applying the significance
%                           critera specified through 'threshold' AND
%                           'consecutive'. Essentially, only the 1's in
%                           bins that pass BOTH criteria are retained, and
%                           1's in bins that fail to pass the criteria are
%                           set to 0, while the 1's in
%                           analysis.unfilteredSigNxB represent bins that
%                           have passed the 'threshold' criteria ALONE.
%                           Note that all futher analyses performed on
%                           "signficant neurons" in this function are based
%                           on analysis.sigNxB. Note as well that
%                           analysis.sigNxB = analysis.unfilteredSigNxB if
%                           the 'consecutive' name-value pair is set to
%                           false or 0.
%   .unfilteredSigNxB    -- nNeuronsPop x nBins matrix with a 1 at the i_th
%                           row j_th column entry if the i_th neuron was
%                           significant for specified predictor during the
%                           j_th bin, 0 otherwise (nNeuronsPop is the total
%                           number of neurons in the population, i.e. with
%                           no regard to significance). The i_th neuron is
%                           deemed significant in the j_th bin if its p
%                           value for the specified predictor is below the
%                           threshold set throught the 'threshold'
%                           name-value pair. There is NO additional
%                           "filtering" here by considering whether neurons
%                           are signficant for a number of consecutive bins
%                           (as specified through the 'consecutive'
%                           name-value pair); this is what distinguishes
%                           this field from the above analysis.sigNxB,
%                           which means that analysis.sigNxB =
%                           analysis.unfilteredSigNxB if the 'consecutive'
%                           name-value pair was set to false or 0.
%   .sigNeuronPredValues -- nNeuronsPop x 1 vector, where the i_th element
%                           takes on a value of 1 if the i_th neuron was
%                           significant for the specified predictor and
%                           assigned a predictor value of 1 (e.g., positive
%                           valence, left direction), -1 if the i_th neuron
%                           was significant for the specified predictor and
%                           assigned a predictor value of -1 (e.g.,
%                           negative valence, right direction), and NaN if
%                           the neuron was not deemed significant. 
%   .sigNeuronMeanBetas  -- nNeuronsPop x 1 vector, where the i_th element
%                           takes on the mean regression beta coefficient
%                           of the i_th neuron for the specified predictor
%                           (averaged over bins in the analysis window) if
%                           the i_th neuron was significant for the
%                           specified predictor. The i_th element has a NaN
%                           value if the neuron was not deemed significant.
%   .sigNeuronFlip       -- nNeuronsFlip x 1 vector, where nNeuronsFlip is
%                           the number of neurons that are signficant for
%                           both values of the specified predictor during
%                           the analysis window of interest (e.g.,
%                           signficant for both positive and negative
%                           valence). The elements of this vector are the
%                           indices of such neurons.
%   .nSigNeurons         -- Scalar value that is the number of signficant
%                           neurons (see 'checkSigIn' name-value pair
%                           under PARAMETERS above).
%   .nSigPredValue1      -- Scalar value that is the number of signficant
%                           neurons encoding predictor value 1 (e.g.
%                           positive valence). (see 'checkSigIn' name-value
%                           pair under PARAMETERS above)
%   .nSigPredValue2      -- Scalar value that is the number of signficant
%                           neurons encoding predictor value w (e.g.
%                           negative valence). (see 'checkSigIn' name-value
%                           pair under PARAMETERS above)
%   .nSigPredValue1Idc   -- Indices of neurons signficantly encoding
%                           encoding predictor value 1 (e.g. positive
%                           valence). (see 'checkSigIn' name-value pair
%                           under PARAMETERS above)
%   .nSigPredValue2Idc   -- Indices of neurons signficantly encoding
%                           encoding predictor value w (e.g. negative
%                           valence). (see 'checkSigIn' name-value pair
%                           under PARAMETERS above)
%   .popNeuronPredValues -- nNeurons x 1 vector, where the i_th element
%                           takes on a value of 1 if the i_th neuron was                           
%                           assigned a predictor value of 1 (e.g., positive
%                           valence, left direction), and -1 if the i_th
%                           neuron was assigned a predictor value of -1
%                           (e.g., negative valence, right direction). All
%                           neurons in the population, regardless of
%                           significant status, are assigned a predictor
%                           value here based on the mean beta coefficient
%                           for the specified predictor, averaged over all
%                           bins within the analysis window.
%   .popNeuronMeanBetas  -- nNeurons x 1 vector, where the i_th element
%                           is the i_th neuron's mean beta coefficient for
%                           the specified predictor, averaged over all bins
%                           within the analysis window.
%   .popNeuronFlip       -- nNeuronsPop x 1 vector, whose elements are the
%                           indices of any neurons (without regard to
%                           significance) which are assigned predictor
%                           values of both 1 and -1 (as based on mean
%                           betas, see analsis.popNeuronPredValues) in bins
%                           within the analysis window.
%   .nPopNeurons         -- Scalar value that is the number of neurons in
%                           the whole population, with no regard to
%                           signficance.
%   .nPopPredValue1      -- Scalar value that is the number of
%                           neurons in the whole population, without regard
%                           to significance, encoding predictor value 1
%                           (e.g. positive valence). See
%                           analysis.popNeuronPredValues.
%`  .nPopPredValue2      -- Scalar value that is the number of
%                           neurons in the whole population, without regard
%                           to significance, encoding predictor value -1
%                           (e.g. negative valence). See
%                           analysis.popNeuronPredValues.
%   .popPredValue1Idc    -- Indices of all neurons, without regard to
%                           significance, encoding predictor value 1 (e.g.
%                           positive valence). See
%                           analysis.popNeuronPredValues.
%   .popPredValue2Idc    -- Indices of all neurons, without regard to
%                           significance, encoding predictor value -1 (e.g.
%                           negative valence). See
%                           analysis.popNeuronPredValues.
%   .meanSigCPDs         -- nNeuronsPop x 1 vector, where the i_th element
%                           is the coefficient of partial determination
%                           (CPD) of the i_th neuron, averaged over
%                           signficant bins ONLY within the specified
%                           analysis window (i.e., even for neurons deemed
%                           significant, bins within the analysis window
%                           that fail to meet signficance criteria are
%                           excluded here) if the i_th neurons was deemed
%                           significant, and NaN otherwise.
%   .meanPopCPDs         -- nNeuronsPop x 1 vector, where the i_th element
%                           is the coefficient of partial determination
%                           (CPD) of the i_th neuron, averaged over all
%                           bins within the specified analysis window, for
%                           all neurons in the population.
%   .pRanksumSig         -- P value from a ranksum test comparing the CPDs
%                           (averaged across signficant bisn only, within
%                           the analysis window) of neurons encoding
%                           predictor value 1 vs predictor value 2.
%   .sigRanksum          -- Ranksum (U) from above ranksum test on
%                           signficant CPDs.
%   .sigEffectSize       -- Effect size (difference in medians, in units of
%                           CPDs) from above ranksum test on significant
%                           CPDs.
%   .pRanksumPop         -- P value from a ranksum test comparing the CPDs
%                           (averaged over all bins within the analysis
%                           window) of all neurons encoding predictor value
%                           1 vs predictor value 2, without regard to
%                           significance.
%   .popRanksum          -- Ranksum (U) from above ranksum test on
%                           all population CPDs.
%   .popEffectSize       -- Effect size (difference in medians, in units of
%                           CPDs) from above ranksum test on all population
%                           CPDs.
%   .meanSigCDs          -- nNeuronsPop x 1 vector, where the i_th element
%                           is the coefficient of determination (CD) of the
%                           i_th neuron, averaged over signficant bins ONLY
%                           within the specified analysis window (i.e.,
%                           even for neurons deemed significant, bins
%                           within the analysis window that fail to meet
%                           signficance criteria are excluded here) if the
%                           i_th neurons was deemed significant, and NaN
%                           otherwise. Not to be confused with
%                           analysis.meanSigCPDs (for coefficients of
%                           PARTIAL determination).
%   .meanPopCDs          -- nNeuronsPop x 1 vector, where the i_th element
%                           is the coefficient of determination (CD) of the
%                           i_th neuron, averaged over all bins within the
%                           specified analysis window, for all neurons in
%                           the population. Not to be confused with
%                           analysis.meanPopCPDs (for coefficients of
%                           PARTIAL determination).
%   .predictor           -- String value that is the predictor of interest
%                           specified through the predictor argument. This
%                           is included primarily for clarity/housekeeping
%                           when looking at the output struct, and for
%                           reference in downstream functions.
%
% Author: Jonathan Chien Version 2.0. 6/29/21.


arguments
    regression
    predictor string 
    nvp.epochBounds (1,2) = [-999 1500]
    nvp.analysisWindowBounds (1,2) = [101 600]
    nvp.binWidth (1,1) = 150
    nvp.sliding = true
    nvp.step (1,1) {mustBeInteger} = 25 
    nvp.threshold (1,1) {mustBeNumeric} = 0.01
    nvp.consecutive = 3
    nvp.checkSigIn string = 'analysisWindow' % 'analysisWindow' or 'epoch'
end


%% Calculate parameters

% Get total number of neurons and bins, as well as bin indices
% corresponding to analysis window of interest (specified in 'window').
nNeurons = size(regression.betaVal, 1);
[epochTimepoints,~,nBins] = get_bin_timepoints('window', nvp.epochBounds, ...
                                               'binWidth', nvp.binWidth, ...
                                               'sliding', nvp.sliding, ...
                                               'step', nvp.step, ...
                                               'wrt', 'event');
[windowTimepoints,~,~] = get_bin_timepoints('window', nvp.analysisWindowBounds, ...
                                            'binWidth', nvp.binWidth, ...
                                            'sliding', nvp.sliding, ...
                                            'step', nvp.step, ...
                                            'wrt', 'event');
windowBinIdc = ismember(epochTimepoints, windowTimepoints); % logical vector


%% Check inputs against significance criteria

% First, obtain an nNeurons x nBins of p values and beta coefficients for
% the specified predictor.
switch predictor
    case 'valence'
        pvalues = regression.pVal;
        betas = regression.betaVal;  
        cpds = regression.cpdVal;
    case 'direction'
        pvalues = regression.pDir;
        betas = regression.betaDir;
        cpds = regression.cpdDir;
    case 'interaction'
        pvalues = regression.pInt;
        betas = regression.betaInt;
        cpds = regression.cpdInt;
    otherwise
        error("Invalid or unsupported value for 'predictor' was passed in.")
end

% Create nNeurons x nBins matrix with a 1 at the i_th row j_th column entry
% if the j_th bin was significant for the i_th neuron. For all entries = 1,
% use the corresponding beta (i_th neuron in the j_th bin) to retain
% predictor value 1 (e.g., positive for valence, left for direction) as 1
% if beta > 0, but turn predictor value 2 (e.g., negative for valence,
% right for direction) into -1 if beta < 0.
unfilteredSigNxB = double(pvalues < nvp.threshold);
unfilteredSigNxB(pvalues < nvp.threshold & betas < 0) = -1;

% If desired, filter out any 1's not part of a consecutive cluster (of
% length specified through the 'consecutive' name-value pair).
if nvp.consecutive
    
    % Preallocate zeros matrix, with 1's to be placed at entries with
    % clusters of signficant bins.
    filteredSigNxB = zeros(nNeurons, nBins);
    
    % Iterate over neurons, for each filtering out bins not part of a
    % consecutive series. In the following inline comments, b =
    % nvp.consecutive for brevity.
    for iNeuron = 1:nNeurons
        iBin = 1;
        while iBin <= nBins
            % If current and following b consecutive bins all = 1.
            if iBin <= nBins-(nvp.consecutive-1) ...
               && sum(unfilteredSigNxB(iNeuron,iBin:iBin+nvp.consecutive-1)) ...
                  == nvp.consecutive                
                filteredSigNxB(iNeuron,iBin:iBin+nvp.consecutive-1) ...
                    = ones(1,nvp.consecutive);
                iBin = iBin + nvp.consecutive;
                
            % If current and preceding b consecutive bins all = 1. This
            % allows us to grow a series of significant bins past the
            % minimum (e.g. if b bins = 3, but there is a series of 5
            % significant bins in a row, this will be detected).
            elseif iBin >= nvp.consecutive ...
                   && sum(unfilteredSigNxB(iNeuron,iBin-(nvp.consecutive-1):iBin)) ...
                      == nvp.consecutive                
                filteredSigNxB(iNeuron,iBin) = 1;
                iBin = iBin + 1;
                
            % If current and following b consecutive bins all = -1.    
            elseif iBin <= nBins-(nvp.consecutive-1) ...
                   && sum(unfilteredSigNxB(iNeuron,iBin:iBin+nvp.consecutive-1)) ...
                       == -nvp.consecutive                
                filteredSigNxB(iNeuron,iBin:iBin+nvp.consecutive-1) ...
                    = -ones(1,nvp.consecutive);
                iBin = iBin + nvp.consecutive;
            
            % If current and preceding b consecutive bins all = -1. See
            % inline comment for second condition above for reasoning.
            elseif iBin >= nvp.consecutive ...
                   && sum(unfilteredSigNxB(iNeuron,iBin-(nvp.consecutive-1):iBin)) ...
                      == -nvp.consecutive                
                filteredSigNxB(iNeuron,iBin) = -1;
                iBin = iBin + 1;
            
            % If no series of b significant bins, beginning from current
            % bin, is detected.
            else
                iBin = iBin + 1;
            end           
        end        
    end
    
    % Assign contents of filteredSigNxB to variable sigNxB (so rest of
    % function can reference the same thing regardless of the value of
    % nvp.consecutive.
    sigNxB = filteredSigNxB;
else
    % Assign contents of unfilteredSigNxB to variable sigNxB (so rest of
    % function can reference the same thing regardless of the value of
    % nvp.consecutive.
    sigNxB = unfilteredSigNxB;
end

% Place in struct to be returned.
analysis.sigNxB = sigNxB;
if nvp.consecutive, analysis.unfilteredSigNxB = unfilteredSigNxB; end


%% Assign predictor value to significant neurons 

% Preallocate for subsequent loop.
sigNeuronPredValues = NaN(nNeurons, 1);
sigNeuronMeanBetas = NaN(nNeurons, 1);
sigNeuronFlip = [];

% For all neurons deemed significant (see first inline comment in loop
% below), take average beta values for specified predictor over all bins in
% the analysis window and use these mean values to assign a single value to
% each neuron overall (across the analysis window) with respect to the
% predictor (e.g., each significant neuron is 'positive' or 'negative' for
% the valence predictor, denoted as +1 or -1, respectively).
for iNeuron = 1:nNeurons
    
    % Ensure a valid value was provided for nvp.checkSigIn, else if
    % statement below will fail to detect any significant neurons.
    assert((strcmp(nvp.checkSigIn, 'analysisWindow') ...
            || strcmp(nvp.checkSigIn, 'epoch')), ...
            "Invalid value for 'checkSigIn'.")
        
    % For current neuron (row of sigNxB), check for any nonzero elements,
    % either over the entire epoch or within the analysis window (as
    % specified through the 'checkSigIn' nvp). If filtering has been
    % performed, any nonzero elements must be clustered, and nonzero rows
    % may be considered significant. If no filtering was performed, we
    % consider even one significant bin enough to deem a neuron significant
    % (and again, need only check for nonzero elements in the current row).
    if (strcmp(nvp.checkSigIn, 'analysisWindow') && any(sigNxB(iNeuron,windowBinIdc))) ...
       || (strcmp(nvp.checkSigIn, 'epoch') && any(sigNxB(iNeuron,:)))

        % Calculate mean beta across all bins within analysis window. Note
        % that we calculate mean betas only over analysis window,
        % regardless of value of 'checkSigIn'.
        meanWindowBeta = mean(betas(iNeuron,windowBinIdc));
        sigNeuronMeanBetas(iNeuron) = meanWindowBeta;

        % Assign predictor value to signficant neuron based on mean
        % beta within analysis window.
        if meanWindowBeta > 0
            sigNeuronPredValues(iNeuron) = 1;
        elseif meanWindowBeta < 0
            sigNeuronPredValues(iNeuron) = -1;
        elseif meanWindowBeta == 0
            warning(['Mean beta coefficients over all bins in ' ...
                     'analysis window for neuron ' num2str(iNeuron) ...
                     'equals zero. No ' predictor ' value assigned.'])
        end
        
        % Check to see if current significant neuron flips encoding (e.g.,
        % from positive to negative valence) DURING ANALYSIS WINDOW.
        if any(sigNxB(iNeuron,windowBinIdc)>0) ...
           && any(sigNxB(iNeuron,windowBinIdc)<0)
            sigNeuronFlip = [sigNeuronFlip; iNeuron];
        end
    end       
    
end

% Place in struct to be returned.
analysis.sigNeuronPredValues = sigNeuronPredValues;
analysis.sigNeuronMeanBetas = sigNeuronMeanBetas;
analysis.sigNeuronFlip = sigNeuronFlip;
analysis.nSigNeurons = sum(~isnan(sigNeuronPredValues));
analysis.nSigPredValue1 = sum(sigNeuronPredValues==1);
analysis.nSigPredValue2 = sum(sigNeuronPredValues==-1);
analysis.sigPredValue1Idc = find(sigNeuronPredValues==1);
analysis.sigPredValue2Idc = find(sigNeuronPredValues==-1);


%% Assign predictor value for all neurons

% Preallocate for subsequent loop.
popNeuronPredValues = NaN(nNeurons, 1);
popNeuronMeanBetas = NaN(nNeurons, 1);
popNeuronFlip = [];

% Despite some overlap, this is done in a separate loop from the above
% section due to indexing (goes 1:nBins - (nvp.consecutive-1) above) and
% for additional clarity.
for iNeuron = 1:nNeurons
    
    % Calculate mean beta across all bins within analysis window.
    meanWindowBeta = mean(betas(iNeuron,windowBinIdc));
    popNeuronMeanBetas(iNeuron) = meanWindowBeta;
    
    % Assign predictor value to signficant neuron based on mean beta within
    % analysis window.
    if meanWindowBeta > 0
        popNeuronPredValues(iNeuron) = 1;
    elseif meanWindowBeta < 0
        popNeuronPredValues(iNeuron) = -1;
    elseif meanWindowBeta == 0
        warning(['Mean beta coefficients over all bins in ' ...
                 'analysis window for neuron ' num2str(iNeuron) ...
                 'equals zero. No ' predictor ' value assigned.'])
    end
    
    % Check to see if current population neuron flips encoding (e.g.,
    % from positive to negative valence) DURING ANALYSIS WINDOW.
    if any(betas(iNeuron,windowBinIdc)>0) ...
       && any(betas(iNeuron,windowBinIdc)<0)
        popNeuronFlip = [popNeuronFlip; iNeuron];
    end
end

% Place in struct to be returned. 
analysis.popNeuronPredValues = popNeuronPredValues;
analysis.popNeuronMeanBetas = popNeuronMeanBetas;
analysis.popNeuronFlip = popNeuronFlip;
analysis.nPopNeurons = length(popNeuronPredValues);
analysis.nPopPredValue1 = sum(popNeuronPredValues==1);
analysis.nPopPredValue2 = sum(popNeuronPredValues==-1);
analysis.popPredValue1Idc = find(popNeuronPredValues==1);
analysis.popPredValue2Idc = find(popNeuronPredValues==-1);


%% Get mean coeff of partial determination for sig and pop neurons
% Averaged over analysis window.

sigCPDs = cpds;
sigCPDs(sigNxB==0) = NaN;
analysis.meanSigCPDs = mean(sigCPDs(:,windowBinIdc), 2, 'omitnan');
analysis.meanPopCPDs = mean(cpds(:,windowBinIdc), 2);

% Run rank sum test between the two predictor values for significant
% neurons, store p value and ranksum and calculate effect size. Note that
% if there are no neurons encoding one or both of the predictor values, no
% ranksum can be calculated for sig neurons (this is handled by the
% try-catch block).
try 
    [analysis.pRanksumSig,~,sigStats] ...
        = ranksum(analysis.meanSigCPDs(sigNeuronPredValues==1), ...
                  analysis.meanSigCPDs(sigNeuronPredValues==-1));
    analysis.sigRanksum = sigStats.ranksum;
    analysis.sigEffectSize ...
        = median(analysis.meanSigCPDs(sigNeuronPredValues==1)) ...
          - median(analysis.meanSigCPDs(sigNeuronPredValues==-1));
catch
    warning(['There may be no significant neurons encoding one of the ' ...
             'predictor values, or no ' newline 'significant neurons ' ...
             'encoding either of the predictor values. Cannot calculate ' ...
             'rank sum for ' newline 'signficant neurons.'])
   analysis.pRanksumSig = [];
   analysis.sigRanksum = [];
   analysis.sigEffectSize = [];
end

% As above, run rank sum test between the two predictor values, but now for
% all neurons. Store p value and ranksum and calculate effect size.
[analysis.pRanksumPop,~,popStats] ...
    = ranksum(analysis.meanPopCPDs(popNeuronPredValues==1), ...
              analysis.meanPopCPDs(popNeuronPredValues==-1));
analysis.popRanksum = popStats.ranksum;
analysis.popEffectSize ...
    = median(analysis.meanPopCPDs(popNeuronPredValues==1)) ...
      - median(analysis.meanPopCPDs(popNeuronPredValues==-1));


%% Get coeff of determination for significant neurons; mean for sig and pop

% Get nNeurons x nBins matrix of coefficients of determination. Also form
% an nNeurons x nBins matrix with values only where bins are significant.
popCDs = regression.rSqr;
sigCDs = regression.rSqr;
sigCDs(sigNxB==0) = NaN;

% For each neuron, average across bins within analysis window. Do this
% using all bins in all neurons, as well as only significant bins (note
% that if unclustered sig bins were filtered, this will also amount to only
% considering significant neurons).
meanSigCDs = mean(sigCDs(:,windowBinIdc), 2, 'omitnan');
meanPopCDs = mean(popCDs(:,windowBinIdc), 2);

% Place in struct to be returned.
analysis.meanSigCDs = meanSigCDs;
analysis.meanPopCDs = meanPopCDs;

% Return predictor as well.
analysis.predictor = predictor;


end
