function stats = rsa(T, templateInd, nv)
% Accepts a b x c x n array, where b = number of time bins, c = number of
% condition patterns, and n = dimensionality of each condition pattern
% (i.e., number of neurons here). For each c x n slice, this function
% computes the c x c correlation matrix of condition patterns. The c^2 x b
% matrix Y has as its i_th column the flattened c x c correlation matrix from
% the i_th bin. X is a c^2 x p design matrix whose i_th column contains a
% flattened template correlation matrix. We then solve the OLS problem X *
% B_hat + E = Y.
%
% PARAMETERS
% ----------
% T           : b x c x n array of mean firing rates, where b = number of 
%               bins, c = number of conditions, and n = number of neurons.
% templateInd : Vector of positive integers indexing the correlation
%               templates to be used in forming the predictor matrix.
% Name-Value Pairs (nv)
%   'normalize' 
%   'showTemplates' : (1 | 0 (default)), specify whether or not to plot
%                     unflattened template matrices.
%   'colormap'      : ('cool' (default) | string), provide string name of
%                     colormap recognized by MATLAB.
%   'condLabels'    : ([] (default) | c x 1 cell array), provide cell array
%                     of names of conditions, or pass an empty array to
%                     skip this.
%   'pval'          : (1 (default) | 0), specify whether or not to compute
%                     p values for beta coefficients.
%   'nPerms'        : (10000 (default) | positive integer), specify number
%                     of permutations to use to approximate null
%                     distribution.
%
% RETURNS
% -------
% stats : scalar struct with the following fields:
%   .beta      : p x b matrix of beta coefficients, whose i_th j_th
%                element is the weight of the i_th template toward the
%                correlation matrix in the j_th bin. 
%   .predicted : c^2 x b matrix whose j_th column is the optimal linear
%                predictor for the j_th bin. Note that the
%                j_th column of this matrix is also the projection of the
%                j_th column of Y into the column space of X.
%   .resid     : c^2 x b matrix whose j_th column is the error vector
%                (vector rejection) from the projection of the j_th column
%                of Y onto the column space of X. The elements of the j_th
%                column are the residuals for the j_th bin.
%   .ssr       : 1 x b vector of sum of squared residuals. The j_th
%                element is the sum of the squared elements of the j_th
%                column of .resid.
%   .sst       : 1 x b vector of total sum of squares. The j_th element is
%                the sum of the squared components of the j_th column of Y
%                after centering the mean at 0.
%   .cd        : 1 x b vector whose j_th element is the coefficient of
%                determination of the model for the j_th bin (the squared
%                correlation between the j_th bin correlation matrix and
%                its predicted value under the model, also interpretable as
%                the squared cosine similarity of the mean-centered
%                versions of the j_th column of Y and its projection onto
%                the column space of X).
%   .cpd       : p x b matrix of coefficients of partial determination,
%                where the i_th,j_th element is the coefficient for the
%                i_th predictor toward the j_th bin correlation matrix.
%   .p         : Optional p x b matrix of p values, where the i_th,j_th
%                element is the p value attaching to the beta coefficient
%                of the i_th template toward the j_th bin correlation
%                matrix (i.e., the i_th,j_th element of the betas matrix
%                (see above)).
%   .vif       : p x 1 vector whose i_th element is the variance inflation
%                factor (VIF) for the i_th predictor (no value for the
%                intercept term is returned).
%   .cpdPval   : p x b matrix of p values where the i_th,j_th element is
%                the p value attaching to the i_th,j_th element of the
%                matrix in the .cpd field.
%
% Author: Jonathan Chien. 11/2021.

arguments
    T
    templateInd
    nv.normalize = 'bins_x_conditions'
    nv.showTemplates = false % Option to plot heatmap of template regressor matrices
    nv.colormap = 'cool' % colormap options for displayed templates (see MATLAB's colormap documentation)
    nv.condLabels = [] % Can supply cell array of condition labels
    nv.pval = true
    nv.nPerms = 10000
end


% Option to normalize each neuron across conditions, separately for each
% bin.
if strcmp(nv.normalize, 'neurons'), T = zscore(T, 0, 2); end

% Option to normalize each neuron across all conditions and bins
% simultaneously.
if strcmp(nv.normalize, 'bins_x_conditions'), T = zscore(T, 0, [1 2]); end

% Determine number of time bins, number of conditions, and number of correlations.
[nBins, nConds, ~] = size(T); nCorr = nConds^2;

% Create predictors from template correlation matrices.
X = construct_design_mat(nConds, templateInd, ...
                         nv.showTemplates, nv.colormap, nv.condLabels);
nPredictors = size(X, 2);

% Create regression response vector for each bin by unwinding correlation
% matrix. Store these vectors in the columns of "correlations".
Y = NaN(nCorr, nBins);
for iBin = 1:nBins
    CxN = squeeze(T(iBin,:,:));
    corrMat = corr_mat(CxN', CxN');
    Y(:,iBin) = corrMat(:);
end
    
% Find OLS solution across all bins. 
stats = ols(X, Y, ...
            'intercept', true, 'cpd', true, 'pval', true, 'vif', true);

% Optionally compute p values for CPDs across all bins.
if nv.pval

cpdsPerm = NaN(nPredictors, nBins, nv.nPerms);

parfor iPerm = 1:nv.nPerms  
    % Permute correlations in each bin.
    Y_perm = NaN(nCorr, nBins);
    for iBin = 1:nBins
        Y_perm(:,iBin) = Y(randperm(nCorr),iBin);
    end

    % Fit OLS model for permuted response variables. Save only the CPDs.
    statsPerm = ols(X, Y_perm, ...
                 'intercept', true, 'cpd', true, 'pval', false, 'vif', false);
    cpdsPerm(:,:,iPerm) = statsPerm.cpd;
end

% Calculate right-tailed p value for CPD with correction (+1) for random
% permutations.
stats.cpdPval = tail_prob(stats.cpd, cpdsPerm, 3, 'exact', false, 'type', 'right-tailed');

end

end


% --------------------------------------------------
function designMat = construct_design_mat(nConds, templateInd, showTemplates, cmap, condLabels)
% Returns a matrix with flattened template matrices in columns.

% Check that the model has a supported number of conditions.
assert(nConds == 8 || nConds == 12, ...
       ['This function currently only supports 8 conditions ' ...
        '(feedback) or 12 conditions (stimulus and feedback combined)'])
           
% Preallocate.
designMat = NaN(nConds^2, 10);
templates = cell(16,1);

% Template 1: context
templates{1} = [ones(nConds/2, nConds/2); -ones(nConds/2, nConds/2)];
templates{1} = [templates{1} flip(templates{1}, 1)];
designMat(:,1) = templates{1}(:);

% Template 2: response direction
templates{2} = [ones(nConds/4,nConds/4); -ones(nConds/4,nConds/4)];
templates{2} = [templates{2} flip(templates{2}, 1)];
templates{2} = repmat(templates{2}, 2, 2);
designMat(:,2) = templates{2}(:);

% Template 3: context x response direction
templates{3} = templates{1} .* templates{2}; % needed if plotting templates
designMat(:,3) = designMat(:,1) .* designMat(:,2);

% Template 4: abstract context over direction only (valid in both
% epochs).
modelVecsA = [];
for i = 1:nConds/2, modelVecsA = [modelVecsA repmat(randn(10,1), 1, 2)]; end % 10 just to extra safe, though 3 should probably be fine
modelVecsB = [flip(modelVecsA(:,1:nConds/2),2) flip(modelVecsA(:,nConds/2+1:nConds),2)];
templates{4} = corr_mat(modelVecsA, modelVecsB);
templates{4}(~ismembertol(templates{4}, 1, 1e-12)) = 0;
designMat(:,4) = templates{4}(:);

% Template 5: abstract context over relative feedback valence only
% (within a given stimulus valence, this is simply abstraction over
% feedback). Valid for feedback epoch only.
if nConds == 8
    modelVecsA = [repmat(randn(10,2), 1, 2) repmat(randn(10,2), 1, 2)]; 
    modelVecsB = [flip(modelVecsA(:,1:4), 2) flip(modelVecsA(:,5:8), 2)];
    templates{5} = corr_mat(modelVecsA, modelVecsB);
    templates{5}(~ismembertol(templates{5},1,1e-12)) = 0;
    designMat(:,5) = templates{5}(:);
else
    designMat(:,5) = NaN(16, 1);
end

% Template 6: abstract direction over context only.
templates{6} = flip(templates{4}, 1);
designMat(:,6) = templates{6}(:);

% Template 7: abstract direction over relative feedback valence only.
modelVecsA = [randn(10,2) randn(10,2)];
modelVecsB = [flip(modelVecsA(:,1:2),2) flip(modelVecsA(:,3:4),2)];
templates{7} = corr_mat(modelVecsA, modelVecsB);
templates{7}(~ismembertol(templates{7},1,1e-12)) = 0;
templates{7} = repmat(templates{7}, 2, 2);
designMat(:,7) = templates{7}(:);

% Template 8: abstract context over direction x abstrct direction over context.
templates{8} = templates{4} .* templates{5}; % needed if plotting templates
designMat(:,8) = designMat(:,4) .* designMat(:,5);

% Template 9: relative feedback valence.
if nConds == 8
    modelVecs = repmat(randn(10,1), 1, 8);
    modelVecs(:,[2 4 6 8]) = modelVecs(:,[2 4 6 8]) * -1;
    templates{9} = corr_mat(modelVecs, modelVecs);
    designMat(:,9) = templates{9}(:);
else
    designMat(:,9) = NaN(16, 1);
end
    
% Template 10: feedback (0 bar vs bar change), where all 0 bar
% situations are modeled as correlated, and all bar change conditions
% (+1 bar and -1 bar conditions) are correlated.
if nConds == 8
    modelVecs = repmat(randn(10,2), 1, 2);
    modelVecs = [modelVecs flip(modelVecs, 2)];
    templates{10} = corr_mat(modelVecs, modelVecs);
    templates{10}(~ismembertol(templates{10},1,1e-12)) = 0;
    designMat(:,10) = templates{10}(:);
else
    designMat(:,10) = NaN(16, 1);
end

% Template 11: 0 bar is modeled as correlated with itself; everything
% else is modeled as uncorrelated.
if nConds == 8
    zeroBarCond = repmat(randn(10,1), 1, 4);
    modelVecs = randn(10,8);
    modelVecs(:,[2 4 5 7]) = zeroBarCond;
    templates{11} = corr_mat(modelVecs, modelVecs);
    templates{11}(~ismembertol(templates{11},1,1e-12)) = 0;
    designMat(:,11) = templates{11}(:);
else
    designMat(:,11) = NaN(16, 1);
end

% Template 12: +1 bar, 0 bar, and -1 bar each correlated with
% themselves and themselves only. The +1 bar and -1 bar contingencies
% are generally correlated with stimulus valence, but here we are
% modeling them as only correlated with other conditions that also
% share the same bar change (but with the other direction). E.g.,
% +pic/+bar/L is correlated only with +pic/+bar/R, and not with
% +pic/0bar/L or +pic/0bar/R, even those these latter two conditions
% also share the +pic attribute.
posBarVec = repmat(randn(10,1), 1, 2);
zeroBarVec = repmat(randn(10,1), 1, 4);
negBarVec = repmat(randn(10,1), 1, 2);
modelVecs = randn(10,8);
modelVecs(:,[1 3]) = posBarVec;
modelVecs(:,[2 4 5 7]) = zeroBarVec;
modelVecs(:,[6 8]) = negBarVec;
templates{12} = corr_mat(modelVecs, modelVecs);
templates{12}(~ismembertol(templates{12},1,1e-12)) = 0;
designMat(:,12) = templates{12}(:);

% Template 13: same as Template 12, but now +1 bar and -1 bar
% conditions are modeled as anticorrelated rather than uncorrelated.
antiCorr = zeros(8, 8);
antiCorr([6 8 22 24 41 43 57 59]) = -1;
templates{13} = templates{12} + antiCorr;
designMat(:,13) = templates{13}(:);

% Template 14:
linearIdc = NaN(4,4);
for iRow = 1:4
    linearIdc(iRow,:) = (1:4) + (8 * (iRow-1));
end
linearIdc = linearIdc(:);
templates{14} = zeros(8,8);
templates{14}(linearIdc) = templates{12}(linearIdc);
designMat(:,14) = templates{14}(:);

% Template 15:
linearIdc = [];
for iRow = 1:4
    linearIdc(iRow,:) = (37:40) + (8 * (iRow-1));
end
linearIdc = linearIdc(:);
templates{15} = zeros(8,8);
templates{15}(linearIdc) = templates{12}(linearIdc);
designMat(:,15) = templates{15}(:);

% Template 16: identity matrix (modeling out correlation of each
% condition with itself.
templates{16} = eye(nConds);
designMat(:,16) = templates{16}(:);

% Ensure that regressors for modeling out the diagonal is present, even
% if not specified manually.
if ~ismember(16, templateInd)
    if iscolumn(templateInd), templateInd = templateInd'; end
    templateInd = [templateInd 16];
end

% Sort template indices to be retained and drop rest from the design
% matrix.
templateInd = sort(templateInd);
designMat = designMat(:,templateInd);

% Option to return heatmaps of retained templates.
if showTemplates
    
    figure
    colormap(cmap)
    templatesToPlot = templates(templateInd);
    nTemplates = length(templateInd);
    harray = gobjects(nTemplates, 1);
    
    for iTemplate = 1:nTemplates
        
        % Determine shape of subplot based on number of templates.
        if nTemplates <= 6
            harray(iTemplate) = subplot(2,3,iTemplate);
        elseif nTemplates <= 9
            harray(iTemplate) = subplot(3,3,iTemplate);
        else
            harray(iTemplate) = subplot(3,5,iTemplate);
        end
        
        % Plot current template.
        imagesc(templatesToPlot{iTemplate})
        caxis(harray(iTemplate), [-1 1])
        colorbar
        title(sprintf('Template %d', templateInd(iTemplate)))
        
        % Label conditions, if condition labels were supplied
        if ~isempty(condLabels)
            set(gca, 'XTick', 1:8, 'XTickLabel', condLabels, ...
                'YTick', 1:8, 'YTickLabel', condLabels);
            xtickangle(45)
        end
    end
    
    sgtitle('Regressor Templates')
end

end


% --------------------------------------------------
function corrMat = corr_mat(A, B)

% Calculates cross correlation matrix for r.v.s as columns of A and B. 

A = A - mean(A);
B = B - mean(B);
corrMat = (A ./ vecnorm(A))' * (B ./ vecnorm(B));
    
end

