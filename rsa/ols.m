function stats = ols(X, Y, nv)
% Fits an ordinary least squares model for an n x p design matrix X, and an
% n x m matrix Y of response variables. If (not counting a possible
% intercept term) p = m = 1 , this is univariate simple regression. If m =
% 1, and p > 1, this is univariate multiple regression. If m > 1 and p = 1,
% this is multivariate simple regression. If both m > 1 and p > 1, this is
% multivariate multiple regression. For all cases, the OLS solution
% consists of a p x m array of beta coefficients whose i_th j_th element
% is the weight of the i_th predictor toward the j_th response variable.
% This function uses the mldivide syntax, which based on the nonsquare
% array shapes here will direct MATLAB to use its QR solver for a fast
% and stable analytical solution.
%
% PARAMETERS
% ----------
% X -- n x p design matrix featuring n observations and p predictors. If
%      p = 1 (not counting a possible intercept term), this is simple
%      regression; if p > 1, this is multiple regression. X should not
%      feature a column of ones to account for an intercept term; instead
%      see the 'intercept' name-value pair.
% Y -- n x m matrix of response variables, where the i_th,j_th element is
%      the value of the j_th dependent variable on the i_th 
%      observation/trial. If m = 1, this is univariate regression; if m >
%      1, this is multivariate regression.
% Name-Value Pairs (nv)
%   'intercept' -- (1 (default) | 0). Specify whether or not to append a
%                  column of ones to the design matrix to allow for a
%                  nonzero intercept in the model. Note that if so, this
%                  column (and any associated p values or CPDs) comes last.
%   'cpd'       -- (1 (default) | 0 | 'include_intercept'). If true,
%                  coefficients of partial determination (CPD) will be
%                  computed for each predictor, not including a possible
%                  intercept term, over all response variables. If
%                  'include_intercept', CPDs will be computed for the
%                  intercept as well. If false, computation of CPDs will be
%                  suppressed.
%   'pval'      -- (1 (default)| 0). Specify whether or not to compute
%                  p-values for beta coefficients. 
%   'vif'       -- (1 (default)| 0). Specify whether or not to compute
%                  the variance inflation factors (VIF) for each of the
%                  predictors (not including any potential intercept).
% 
% RETURNS
% -------
% stats -- Scalar struct with regression results in the following fields:
%   .beta      -- p x m matrix of beta coefficients, whose i_th j_th
%                 element is the weight of the i_th predictor toward the
%                 j_th response variable. Note that if a nonzero intercept
%                 term was included, the last row of p corresponds to this
%                 term.
%   .predicted -- n x m matrix whose j_th column is the optimal linear
%                 predictor for the j_th response variable. Note that the
%                 j_th column of this matrix is also the projection of the
%                 j_th column of Y into the column space of X.
%   .resid     -- n x m matrix whose j_th column is the error vector
%                 (vector rejection) from the projection of the j_th column
%                 of Y onto the column space of X. The elements of the j_th
%                 column are the residuals for the j_th response variable.
%   .ssr       -- 1 x m vector of sum of squared residuals. The j_th
%                 element is the sum of the squared elements of the j_th
%                 column of .resid.
%   .sst       -- 1 x m vector of total sum of squares. The j_th element is
%                 the sum of the squared components of the j_th column of Y
%                 after centering the mean at 0.
%   .cd        -- 1 x m vector whose j_th element is the coefficient of
%                 determination of the model for the j_th response variable
%                 (the squared correlation between the j_th response
%                 variable and its predicted value under the model, also
%                 interpretable as the squared cosine similarity of the
%                 mean-centered versions of the j_th column of Y and its
%                 projection onto the column space of X).
%   .cpd       -- Optional p x m matrix of coefficients of partial
%                 determination, where the i_th,j_th element is the
%                 coefficient for the i_th predictor toward the j_th
%                 response variable (there is no coefficient here for the
%                 intercept term by default, but the user can request one
%                 by setting 'cpd' to 'includeIntercept' (see PARAMETERS),
%                 in which case the last row will contain CPDs for the
%                 intercept, with the j_th column element corresponding to
%                 the j_th response variable). If 'cpd' = false, this field
%                 will be absent from the returned stats struct.
%   .p         -- Optional p x m matrix of p values, where the i_th,j_th
%                 element is the p value attaching to the beta coefficient
%                 of the i_th predictor toward the j_th response variable
%                 (i.e., the i_th,j_th element of the betas matrix (see
%                 above)). If 'pval' = false (see PARAMETERS), this field
%                 will be absent from the returned stats struct.
%   .vif       -- p x 1 vector whose i_th element is the variance inflation
%                 factor (VIF) for the i_th predictor (no value for the
%                 intercept term is returned, if one was included, as
%                 calculation of VIFs here requires the correlation matrix
%                 of X). If 'vif' = false (see PARAMETERS), this field will
%                 be absent from the returned struct.
%
% Author: Jonathan Chien 1/12/22. Last edit: 2/4/22.

arguments
    X
    Y
    nv.intercept = true
    nv.cpd = true
    nv.pval = true
    nv.vif = true
end

% Check array sizes and get number of dependent variables.
n_obs = size(X, 1);
assert(n_obs == size(Y, 1), 'First array dim sizes of X and Y must be equal.')
n_dv = size(Y, 2); 

% Optionally add vector of ones for intercept term. Get number of columns
% of X after possibly adding ones; define nIv (number of independent vars
% as number of columns of X not including intercept, if one was included).
assert(~any(sum(X) == n_obs), ...
       "Do not pass in X with a column of ones. If an intercept term is " + ...
       "desired, set 'intercept' to true.")
if nv.intercept, X = [X ones(n_obs, 1)]; end; n_cols = size(X, 2); 
if nv.intercept, n_iv = n_cols - 1; else, n_iv = n_cols; end 

% Fit OLS model.
stats.beta = X \ Y; 
stats.predicted = X * stats.beta;
stats.resid = Y - stats.predicted;
stats.ssr = sum( (stats.resid).^2 );

% Calculate coefficient of determination for all dependent variables. 
stats.sst = sum( (Y - mean(Y)).^2 );
stats.cd = 1 - (stats.ssr ./ stats.sst);

% Calculate coefficient of partial determination for each predictor (not
% including intercept by default), for each dependent variable.
if nv.cpd
    if strcmp(nv.cpd, 'include_intercept')
        k = n_cols;
        if n_cols == n_iv
            warning(['A CPD was requested for the intercept term, but ' ...
                     'there is no intercept.'])
        end
    else 
        k = n_iv;
    end
    
    stats.cpd = NaN(k, n_dv);
    for i_iv = 1:k
        X_red = X(:,setdiff(1:n_cols,i_iv));
        stats_red.beta = X_red \ Y;
        stats_red.predicted = X_red * stats_red.beta;
        stats_red.resid = Y - stats_red.predicted;
        stats_red.ssr = sum( (stats_red.resid).^2 );
        stats.cpd(i_iv,:) = (stats_red.ssr - stats.ssr ) ./ stats_red.ssr;
    end
end

% Optionally attach p values to beta cofficients. 
if nv.pval
    dof = n_obs - n_cols; 
    sigmas = stats.ssr / dof;
    se = sqrt(diag(inv(X'*X)) .* sigmas); % .* is outer product in multivariate case
    t = stats.beta ./ se;
    stats.p = 2 * tcdf(abs(t), dof, 'upper');
end

%  Optionally calculate VIF for each of the predictors.
if nv.vif, stats.vif = diag(inv(corrcoef(X(:,1:n_iv)))); end

end
