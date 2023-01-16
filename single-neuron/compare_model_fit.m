function comparison = compare_model_fit(regression1,regression2,analysis1,analysis2,nvp)
% First checks if a neuron is assigned as signficant for stimulus valence
% OR absolute feedback valence, and then uses metric etc. to make a final
% assignment if a neuron is significant for both.

arguments
   regression1 % stimulus valence
   regression2 % absolute feedback valence
   analysis1
   analysis2
   nvp.comparisonMethod = 'rSqr' % ('rSqr' | 'beta_coeff' | 'cpd')
   nvp.epochBounds = [-999 1500]
   nvp.analysisWindowBounds = [101 600]
   nvp.binWidth = 150
   nvp.sliding = true
   nvp.step = 25
   nvp.plot = true % 'filtered', 'unfiltered', or false
   nvp.figure = true
   nvp.legendNames = {'Stimulus valence', 'Absolute feedback valence'}
end

NPREDS = 6;


%% Identify "tough calls"

% First, ensure number of neurons and bins match between regression runs.
[nNeurons1,nBins1] = size(regression1.betaVal); 
[nNeurons2,nBins2] = size(regression1.betaVal); 

if nNeurons1 == nNeurons2 && nBins1 == nBins2
    nNeurons = nNeurons1;
    clear nNeurons1 nNeurons2 nBins1 nBin2
else 
    error(['Either the number of neurons or the number of bins is ' ...
           'mismatched between the two supplied regression runs.'])
end

% "Tough calls" are neurons that are signficant for both picture valence
% and absolute feedback valence. Warn if there are neurons that seem to be
% selective for positive stimulus valence AND negative absolute feedback,
% or negative stimulus valence AND postiive absolute feedback.
toughCalls.pos = intersect(analysis1.sigPredValue1Idc, analysis2.sigPredValue1Idc);
toughCalls.neg = intersect(analysis1.sigPredValue2Idc, analysis2.sigPredValue2Idc);
toughCalls.mixed = [intersect(analysis1.sigPredValue1Idc, analysis2.sigPredValue2Idc); ...
                    intersect(analysis1.sigPredValue2Idc, analysis2.sigPredValue1Idc)];
if ~isempty(toughCalls.mixed)
    disp(['There seem to be neurons that are signficantly selective for ' ...
          'positive stimulus and negative absolute feedback, or negative ' ...
          'stimulus and positive absolute feedback. Check toughCalls.mixed.'])
end

easyCalls.model1Pos = setdiff(analysis1.sigPredValue1Idc, analysis2.sigPredValue1Idc);
easyCalls.model1Neg = setdiff(analysis1.sigPredValue2Idc, analysis2.sigPredValue2Idc);
easyCalls.model2Pos = setdiff(analysis2.sigPredValue1Idc, analysis1.sigPredValue1Idc);
easyCalls.model2Neg = setdiff(analysis2.sigPredValue2Idc, analysis1.sigPredValue2Idc);


%% Compare models for each neuron in each bin.
% For each bin for each neuron, assign one model as the better one based on
% difference in metric, then tally the number of neurons by model in each
% bin using all bins, as well as only bins with significant difference in
% metric.

% Difference in metric between model 1 and 2.
switch nvp.comparisonMethod
    case 'rSqr'
        comparison.diff = regression1.rSqr - regression2.rSqr;
    case 'beta_coeff'
        comparison.diff = abs(regression1.betaVal) - abs(regression2.betaVal);
    case 'cpd'
        comparison.diff = regression1.cpdVal - regression2.cpdVal;
    case 'dev'
        % This is model 2 - model 1 because here smaller deviance indicates
        % better model fit, unlike with the other 3 values.
        comparison.diff = regression2.dev - regression1.dev;
end


if any(comparison.diff == 0, 'all')
    warning(['Difference in metric between models computed as 0 in ' ...
             'at least one bin. Was the same model passed in twice?'])
end

% Assign a 1 where model 1 is preferred, and a -1 where model 2 is
% preferred. Count number of neurons in each bin for which model 1 is
% preferred; do the same for model2.
comparison.unfiltered.tallies = double(comparison.diff > 0);
comparison.unfiltered.tallies(comparison.diff < 0) = -1;
comparison.unfiltered.model1Count = sum(comparison.unfiltered.tallies == 1); 
comparison.unfiltered.model2Count = sum(comparison.unfiltered.tallies == -1);
comparison.unfiltered.model1Prop = comparison.unfiltered.model1Count/nNeurons;
comparison.unfiltered.model2Prop = comparison.unfiltered.model2Count/nNeurons;


%% Make one overall assignment for each neuron (across entire popultion, i.e. unfiltered wrt task selectivity)

% Get indices of bins in analysis window.
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


% First make one assignment for all neurons based on mean difference in
% metric during the analysis window. This assignment merely describes
% preferrence for model (not preference for value of predictor, which we
% will get from the regression betas of the preferred model).
comparison.model1Pref = find(mean(comparison.diff(:,windowBinIdc),2)>0);
comparison.model2Pref = find(mean(comparison.diff(:,windowBinIdc),2)<0);

% Intersect model preference with task selectivity.
comparison.unfiltered.model1Pos ...
    = intersect(comparison.model1Pref, analysis1.popPredValue1Idc);
comparison.unfiltered.model1Neg ...
    = intersect(comparison.model1Pref, analysis1.popPredValue2Idc);
comparison.unfiltered.model2Pos ...
    = intersect(comparison.model2Pref, analysis2.popPredValue1Idc);
comparison.unfiltered.model2Neg ...
    = intersect(comparison.model2Pref, analysis2.popPredValue2Idc);


%% Filter out neuron/bins that were not consecutively signficant for task variable

% Filter against significance results.
comparison.filtered.tallies = comparison.unfiltered.tallies;
comparison.filtered.tallies(comparison.unfiltered.tallies == 1 & abs(analysis1.sigNxB) ~= 1) = 0;
comparison.filtered.tallies(comparison.unfiltered.tallies == -1 & abs(analysis2.sigNxB) ~=1) = 0;
comparison.filtered.tallies = reshape(comparison.filtered.tallies, [nNeurons nBins]);

% Calculate significance counts and proportions based on filtered results.
comparison.filtered.model1Count = sum(comparison.filtered.tallies == 1); 
comparison.filtered.model2Count = sum(comparison.filtered.tallies == -1);
comparison.filtered.model1Prop = comparison.filtered.model1Count/nNeurons;
comparison.filtered.model2Prop = comparison.filtered.model2Count/nNeurons;


%% Make one overall assignment for each signficant neuron (across only neurons sig wrt task selectivity)

% Intersect preference list from entire population with task selectivity
% for signficant neurons.
comparison.filtered.model1Pos ...
    = intersect(comparison.model1Pref, analysis1.sigPredValue1Idc);
comparison.filtered.model1Neg ...
     = intersect(comparison.model1Pref, analysis1.sigPredValue2Idc);
comparison.filtered.model2Pos ...
     = intersect(comparison.model2Pref, analysis2.sigPredValue1Idc);
comparison.filtered.model2Neg ...
     = intersect(comparison.model2Pref, analysis2.sigPredValue2Idc);


%% Store tough and easy calls structs in comparison struct to be returned

comparison.toughCalls = toughCalls;
comparison.easyCalls = easyCalls;


%% Option to plot neurons by preferred model in each bin.
if nvp.plot

    % Ensure valid input.
    assert(any(strcmp(nvp.plot, {'filtered', 'unfiltered'})))

    % Prepare string for title.
    switch nvp.comparisonMethod
        case 'rSqr'
            metric_title = 'R^2';
        case 'beta_coeff'
            metric_title = 'beta coefficients';
        case 'cpd'
            metric_title = 'CPDs';
        case 'dev'
            metric_title = 'deviance';
    end
    
    % Plot count of neurons by preferred model in each bin.
    if nvp.figure
        figure
    else
        subplot(2,1,1)
    end
            
    hold on
    plot(regression1.binCenters, comparison.(nvp.plot).model1Count, 'LineWidth', 2.5, ...
         'Color', [0 0.4470 0.7410], 'DisplayName', nvp.legendNames{1})
    plot(regression2.binCenters, comparison.(nvp.plot).model2Count, 'LineWidth', 2.5, ...
         'Color', [0.6350 0.0780 0.1840], 'DisplayName', nvp.legendNames{2})
    
    title(sprintf('Neuron counts by preferred model (comparison via %s)', metric_title))
    xlabel('Time (ms)')
    ylabel('Count (neurons)')
    legend
    
    % Plot proportion of neurons by preferred model in each bin.
    if nvp.figure
        figure
    else
        subplot(2,1,2)
    end
    
    hold on
    plot(regression1.binCenters, comparison.(nvp.plot).model1Prop, 'LineWidth', 2.5, ...
         'Color', [0 0.4470 0.7410], 'DisplayName', nvp.legendNames{1})
    plot(regression2.binCenters, comparison.(nvp.plot).model2Prop, 'LineWidth', 2.5, ...
         'Color', [0.6350 0.0780 0.1840], 'DisplayName', nvp.legendNames{2})
    
    title(sprintf('Proportions of neurons by preferred model (comparison via %s)', metric_title))
    xlabel('Time (ms)')
    ylabel('Proportion of neurons')
    legend
end

end
