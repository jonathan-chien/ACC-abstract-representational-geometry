function X_red = remove_dims(X, dims)
% Projects rows of a mean-centered data matrix X onto a lower dimensional
% subspace spanned by a subset of the right eigenvectors of the covariance
% matrix/right singular vectors of the mean-centered data matrix. Matrix
% deflation (subtracting outer product) and setting singular values to zero
% and reconstructing are other options.
%
% PARAMETERS
% ----------
% X    : m x n data matrix.
% dims : Positive integers indexing right singular vectors of X and right
%        eigenvectors of X'X.
% 
% RETURNS
% -------
% X_red : m x n data matrix with variance along specified directions set to
%         0.
%
% Author: Jonathan Chien


assert(all(max(dims) < min(size(X, 1)-1, size(X, 2))));

% Subtract mean.
mu = mean(X);
X = X - mu;

% Perform SVD and form projection matrix P.
[~, ~, V] = svd(X);
Q = V(:, setdiff(1:size(V,2),dims));
P = Q*Q';

% Apply vector projection.
X_red = X*P;

% Reapply mean.
X_red = X_red + mu;

end
