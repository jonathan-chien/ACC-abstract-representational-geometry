function comparison ...
    = model_comparison(regressionOut1,regressionOut2,predictor1,predictor2,nvp)
% Takes as input the respective outputs of binned_glm for two regression
% runs (may be on different areas, or different models, etc.), calls
% predictor_analysis2 on each, and compares results.
%
% PARAMETERS
% ----------
% regression1 -- 1 x 1 struct that is the output of binned_glm for one run
%                (i.e., one area, one model, etc.). See the documentation
%                of binned_glm under RETURNS for a complete list of the
%                fields and their respective contents.
% regression2 -- 1 x 1 struct that is the output of binned_glm for one run
%                (one area, one model, etc.; i.e., a different run from the
%                one producing regression1). See the documentation of
%                binned_glm under RETURNS for a complete list of the fields
%                and their respective contents.
% predictor1  -- This argument is a string passed to predictor_analysis2
%                and specifies within that function which predictor we
%                would like to analyze for regression1. Currently the
%                supported values are 'valence' and 'direction'.
% predictor2  -- This argument is a string passed to predictor_analysis2
%                and specifies within that function which predictor we
%                would like to analyze for regression2. Currently the
%                supported values are 'valence' and 'direction'.
% Name-Value Pairs (nvp)
%   'analysisWindowBounds' -- Passed to predictor_analysis2. See that
%                             function's documentation for more info.
%   'epochBounds'          -- Passed to predictor_analysis2. See that
%                             function's documentation for more info.
%   'binWidth'             -- Passed to predictor_analysis2. See that
%                             function's documentation for more info.
%   'sliding'              -- Passed to predictor_analysis2. See that
%                             function's documentation for more info.
%   'step'                 -- Passed to predictor_analysis2. See that
%                             function's documentation for more info.
%   'threshold'            -- Passed to predictor_analysis2. See that
%                             function's documentation for more info.
%   'consecutive'          -- Passed to predictor_analysis2. See that
%                             function's documentation for more info.
%   'checkSigIn'           -- Passed to predictor_analysis2. See that
%                             function's documentation for more info.
%
% RETURNS
% -------
% comparison -- 1 x 1 struct with the results of model comparisons in the
%               following fields:
%   .chiSqr         -- Test statistic from chi-squared test for 2 x 2 table
%                      whose rows and columns correspond, respectively, to
%                      regression runs (regression1 and regression 2) and
%                      predictors (predictor1 and predictor2).
%   .chiDf          -- Number of degrees of freedom for above chi-squared
%                      distribution.
%   .pChi           -- P value for chiSqr under null hypothesis with chiDf
%                      degrees of freedom.
%   .pBinom1        -- For regression1, the p value for a binomial test
%                      under the null hypothesis that the proportions of
%                      neurons significant for predictor1 value 1 (e.g.
%                      positive valence) and predictor1 value 2 (continuing
%                      the example, negative valence) are equal (that is
%                      predictor1 value 1 comprises 0.5 of that significant
%                      neurons, and Bernoulli success = 0.5).
%   .pBinom2        -- For regression2, the p value for a binomial test
%                      under the null hypothesis that the proportions of
%                      neurons significant for predictor2 value 1 (e.g.
%                      positive valence) and predictor2 value 2 (continuing
%                      the example, negative valence) are equal (that is
%                      predictor2 value 1 comprises 0.5 of that significant
%                      neurons, and Bernoulli success = 0.5).
%   .pRanksumSig1   -- For regression1, the p value of the ranksum under
%                      the null hypothesis that the mean (averaged over
%                      analysis window) coefficients of partial
%                      determination (CPDs) of single units signficiant for
%                      predictor1 value1 and predictor1 value 2 have a
%                      difference in median = 0.
%   .sigRanksum1    -- The ranksum whose p value under the null hypothesis
%                      is modelComparison.pRanksumSig1.
%   .sigEffectSize1 -- 
%   .pRanksumSig2   -- For regression2, the p value of the ranksum under
%                      the null hypothesis that the mean (averaged over
%                      analysis window) coefficients of partial
%                      determination (CPDs) of single units signficiant for
%                      predictor2 value1 and predictor2 value 2 have a
%                      difference in median = 0.
%   .sigRanksum2    -- The ranksum whose p value under the null hypothesis
%                      is modelComparison.pRanksumSig2.
%   .pRanksumPop1   -- For regression1, the p value of the ranksum under
%                      the null hypothesis that the mean (averaged over
%                      analysis window) coefficients of partial
%                      determination (CPDs) of all units selective (no
%                      significant criteria applied) for predictor1 value1
%                      and predictor1 value 2 have a difference in median =
%                      0.
%   .ranksumPop1    -- The ranksum whose p value under the null hypothesis 
%                      is modelComparison.pRanksumPop1.
%   .pRanksumPop2   -- For regression2, the p value of the ranksum under
%                      the null hypothesis that the mean (averaged over
%                      analysis window) coefficients of partial
%                      determination (CPDs) of all units selective (no
%                      significant criteria applied) for predictor2 value1
%                      and predictor2 value 2 have a difference in median =
%                      0.
%   .ranksumPop2    -- The ranksum whose p value under the null hypothesis 
%                      is modelComparison.pRanksumPop2.
%  
% Author: Jonathan Chien Version 1.0. 6/25/21. 


arguments
    regressionOut1
    regressionOut2
    predictor1
    predictor2
    nvp.analysisWindowBounds (1,2)
    nvp.epochBounds (1,2) 
    nvp.binWidth = 150
    nvp.sliding = true
    nvp.step = 25
    nvp.threshold = 0.01
    nvp.consecutive = 3
    nvp.checkSigIn = 'analysisWindow'
end


%% Parse inputs and analyze both regression runs

% Rename variables for brevity.
epoch1 = regressionOut1.epoch;
epoch2 = regressionOut2.epoch;
area1 = regressionOut1.area;
area2 = regressionOut2.area;

% Check what is being compared. Warn against comparing between different
% epochs when area and predictor do not both match, and against comparing
% two runs where both area and predictor differ (when both runs are from 
% the same epoch.
if ~strcmp(epoch1,epoch2) ...
   && (~strcmp(area1,area2) || ~strcmp(predictor1,predictor2))
    warning(['Comparison requested between two regression runs from ' ...
             'different epochs but ' newline 'area, predictor or both ' ...
             'do not match.'])
end
if ~strcmp(area1,area2) && ~strcmp(predictor1,predictor2)
    warning(['Comparison requested between two regression runs featuring ' ...
             'both different areas and ' newline 'different predictors.'])
end

% Obtain predictor analysis for both regression runs.
analysis1 = predictor_analysis2(regressionOut1, predictor1, ...
                                'epochBounds', nvp.epochBounds, ...
                                'analysisWindowBounds', nvp.analysisWindowBounds, ...
                                'binWidth', nvp.binWidth, ...
                                'sliding', nvp.sliding, ...
                                'step', nvp.step, ...
                                'threshold', nvp.threshold, ...
                                'consecutive', nvp.consecutive, ...
                                'checkSigIn', nvp.checkSigIn);
                               
analysis2 = predictor_analysis2(regressionOut2, predictor2, ...
                                'epochBounds', nvp.epochBounds, ...
                                'analysisWindowBounds', nvp.analysisWindowBounds, ...
                                'binWidth', nvp.binWidth, ...
                                'sliding', nvp.sliding, ...
                                'step', nvp.step, ...
                                'threshold', nvp.threshold, ...
                                'consecutive', nvp.consecutive, ...
                                'checkSigIn', nvp.checkSigIn);
                               
                               
%% Statistical tests: Chi-squared

% Chi-squared test of number of significant neurons between runs (1's and
% 2's are indices).
chiTable = table([analysis1.nSigPredValue1; analysis2.nSigPredValue1], ...
                 [analysis1.nSigPredValue2; analysis2.nSigPredValue2]);      
[chiSqr,df,p1] = chisq_elr(chiTable);
comparison.chiSqr = chiSqr;
comparison.chiDf = df;
comparison.pChi = p1;


%% Statistical tests: Binomial test (for sig neurons within run)

% Binomial tests for runs 1 and 2 respectively.
comparison.pBinom1 = myBinomTest(analysis1.nSigPredValue1, ...
                                 analysis1.nSigNeurons, ...
                                 0.5);
comparison.pBinom2 = myBinomTest(analysis2.nSigPredValue1, ...
                                 analysis2.nSigNeurons, ...
                                 0.5);
                               
  
%% Statistical tests: Ranksum test between runs (for sig and pop neurons)                        
% Within each predictor value (e.g. positive valence, or left direction),
% perform rank sum comparison of CPDs of neurons between the two runs (CPD
% is averaged over bins of analysis window, for each neuron). This is done
% for both significant neurons and for all neurons. NB: This section
% produces similar output as the function predictor_analysis2 when called
% for a single regression run; however, that function tests between
% predictor values for a single regression run, whereas this function tests
% between regression runs for a single predictor. Thus, we check first to
% ensure the predictor of interest for both runs is the same.

if strcmp(predictor1, predictor2)

% For predictor value 1 (e.g. positive valence or left direction), compare
% between the two runs over sig neurons. Note that if there are no
% significant neurons from one or both runs encoding predictor value 1, the
% ranksum cannot be calculated (try-catch block handles this).
try
    [comparison.pRanksumSig1,~,stats] ...
        = ranksum(analysis1.meanSigCPDs(analysis1.sigPredValue1Idc), ...
                  analysis2.meanSigCPDs(analysis2.sigPredValue1Idc));
    comparison.sigRanksum1 = stats.ranksum;
    comparison.sigEffectSize1 ...
        = median(analysis1.meanSigCPDs(analysis1.sigPredValue1Idc)) ...
          - median(analysis2.meanSigCPDs(analysis2.sigPredValue1Idc));
catch
    warning(['There may be no significant neurons in one of the runs ' ...
             'encoding the predictor value, or no significant neurons in ' ...
             'either of the runs encoding the predictor value. Cannot ' ...
             'calculate rank sum for predictor value 1 signficant neurons.'])
    comparison.pRanksumSig1 = [];
    comparison.sigRanksum1 = [];
    comparison.sigEffectSize1 = [];
end

% For predictor value 2 (e.g., negative valence, or right direction),
% compare between the two runs over sig neurons. As with predictor value 1,
% note that if there are no significant neurons from one or both runs
% encoding predictor value 2, the ranksum cannot be calculated (try-catch
% block handles this).
try
    [comparison.pRanksumSig2,~,stats] ...
        = ranksum(analysis1.meanSigCPDs(analysis1.sigPredValue2Idc), ...
                  analysis2.meanSigCPDs(analysis2.sigPredValue2Idc));
    comparison.sigRanksum2 = stats.ranksum;
    comparison.sigEffectSize2 ...
        = median(analysis1.meanSigCPDs(analysis1.sigPredValue2Idc)) ...
          - median(analysis2.meanSigCPDs(analysis2.sigPredValue2Idc));
catch
    warning(['There may be no significant neurons in one of the runs ' ...
             'encoding the predictor value, or no significant neurons in ' ...
             'either of the runs encoding the predictor value. Cannot ' ...
             'calculate rank sum for predictor value 2 signficant neurons.'])
    comparison.pRanksumSig2 = [];
    comparison.sigRanksum2 = [];
    comparison.sigEffectSize2 = [];
end

% For predictor value 1, compare between the two runs for all neurons.
[comparison.pRanksumPop1,~,stats] ...
    = ranksum(analysis1.meanPopCPDs(analysis1.popPredValue1Idc), ...
              analysis2.meanPopCPDs(analysis2.popPredValue1Idc));
comparison.ranksumPop1 = stats.ranksum;
          
% For predictor value 2, compare between the two runs for all neurons.
[comparison.pRanksumPop2,~,stats] ...
    = ranksum(analysis1.meanPopCPDs(analysis1.popPredValue2Idc), ...
              analysis2.meanPopCPDs(analysis2.popPredValue2Idc));
comparison.ranksumPop2 = stats.ranksum;

% Calculate effect sizes
comparison.popEffectSize1 ...
        = median(analysis1.meanPopCPDs(analysis1.popPredValue1Idc)) ...
          - median(analysis2.meanPopCPDs(analysis2.popPredValue1Idc));
comparison.popEffectSize2 ...
        = median(analysis1.meanPopCPDs(analysis1.popPredValue2Idc)) ...
          - median(analysis2.meanPopCPDs(analysis2.popPredValue2Idc));

end

end
