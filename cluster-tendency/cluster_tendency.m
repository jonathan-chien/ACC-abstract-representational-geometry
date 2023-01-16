function [stats, distr] = cluster_tendency(X, nv)
% Implements ePAIRS as extended by Dubreil et al (2022). Note that the data
% matrix is not mean-centered here, however.
%
% PARAMETERS
% ----------
% X : m x n data matrix. We wish to assess the cluster tendency of the
%     row vectors.
% Name-Value Pairs (nv)
%   'dist'   : ('angular_dist' | string), specify the distance metric to be
%              used for nearest-neighbor distance. Values should correspond
%              to accepted values for the Distance argument to MATLAB's
%              pdist2.m (yeah, yeah, cosine distance ain't a real metric,
%              we know). 
%   'knn'    : (3 | positive integer), specify the number of nearest
%              neighbor distances to average over.
%   'n_null' : (1000 | positive integer), specify the number of draws from
%              the relevant MVN distribution to use to generate a discrete
%              approximation of the null.
%
% RETURNS
% stats : Scalar struct with the following fields:
%   .pairs_stat : PAIRS statistics computed as the (theta_null -
%                 theta_median) / theta_null, where theta_null is the
%                 median of the pooled null mean distances and theta_emp
%                 the median of empirical mean distances.
%   .p          : P value for the ePAIRS test computed from a two-sided
%                 ranksum test of the pooled null mean distances vs the
%                 empirical mean distances.
% distr :  Scalar struct with the following fields:
%   .distances           : m x k array of distances where k is the number
%                          of nearest neighbors specified by the 'knn'
%                          Name-Value Pair and the i_th,j_th element is the
%                          distance from the i_th observation to its j_th
%                          neighbor.
%   .mean_distances      : m x 1 vector whose i_th element is mean distance
%                          of the i_th observation to its k nearest
%                          neighbors.
%   .null_distances      : n x m x k array of distances where n is the
%                          number of draws from the null. The
%                          i_th,j_th,k_th element is the distance of the
%                          j_th element to its k_th neighbor in the i_th
%                          null draw.
%   .null_mean_distances : n x m array where the i_th,j_th element is the
%                          mean of the distances of the j_th element to its
%                          k nearest neighbors in the i_th null draw.
%
% Author: Jonathan Chien.


arguments
    X
    nv.dist = 'angular_dist'
    nv.knn = 3
    nv.n_null = 1000
end


n_obs = size(X, 1);

% Calculate mean vector and covariance matrix of for X
mu = mean(X);
S = cov(X);


%% Nearest neighbor distances of passed in data

distr.distances = calc_distance_to_knn(X, nv.dist, nv.knn);
distr.mean_distances = mean(distr.distances, 2); % Column vector


%% Generate null distribution

null_distances = NaN(nv.n_null, n_obs, nv.knn);

for i_null = 1:nv.n_null    
    % Draw vectors from multivariate normal distribution with mean
    % vector and covariance matrix matching that of the test vectors.
    N = mvnrnd(mu, S, n_obs);

    % Calculate nearest neighbor distances of random vectors. 
    D_null = calc_distance_to_knn(N, nv.dist, nv.knn);

    % Assign into container array.
    null_distances(i_null,:,:) = D_null;
end

distr.null_distances = null_distances;
distr.null_mean_distances = mean(null_distances, 3);


%% Compute PAIRS statistic and p values.

stats.pairs_stat ...
    = (median(distr.null_mean_distances, 'all') - median(distr.mean_distances)) ...
      / median(distr.null_mean_distances, 'all');

stats.p = ranksum(distr.mean_distances, distr.null_mean_distances(:));

end


% --------------------------------------------------
function D = calc_distance_to_knn(X, dist, k)
% X is an m x n matrix of m observations in n dimensions. D is an m x k
% matrix, where k is the number of nearest neighbors. The i_th,j_th element
% is the distance from the i_th observation to its j_th neighbor.

if strcmp(dist, 'angular_dist')
    D = pdist2(X, X, 'cosine', 'Smallest', k+1);
    D(1,:) = [];
    D = real(acos(1 - D)); % Convert to cosine similarity, then to angle
else
    D = pdist2(X, X, dist, 'Smallest', k+1);
    D(1,:) = [];
end

D = D';

end

