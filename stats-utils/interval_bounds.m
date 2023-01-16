function interval = interval_bounds(distr,interval,dim)
% Calculate percentile interval(s) of specified size on null distribution.
% 
% PARAMETERS
% ----------
% distr    -- Array of dimensionality n. Elements along one of the n
%             dimensions (call it the i_th dimension) contain draws from
%             distributions of random variables indexed along the remaining
%             dimensions (i.e., setdiff(1:n, i)). For example, if n = 3,
%             then distr is an k x m x p array, where k and m index
%             variables (e.g., k = number of subjects and m = number of
%             sessions) and p is the number of draws from each
%             distribution.
% interval -- Scalar argument specifying the size of the interval to be
%             calculated for each variable. E.g., set to 95 for a 95%
%             interval.
% dim      -- Scalar nonnegative integer specifying the array dimension
%             along which are the draws from each distribution. In the
%             example under "distr" above, dim should = 3.
%             
% RETURNS
% -------
% interval -- Array of dimensionality n. All dimensions sizes match those
%             of distr, except for the dimension containing the draws from
%             each distribution. In the example mentioned under "distr"
%             (PARAMETERS), interval would be an k x m x 2 array. The first
%             and second elements along the dimension of size 2 are,
%             respectively, the lower and upper bounds of the interval.
%
% Author: Jonathan Chien 8/20/21. Last edit: 12/4/22.

% Check inputs.
assert(dim > 0 && floor(dim) == dim)
assert(isscalar(interval));
assert(interval > 0 && interval < 100);
assert(~isscalar(distr));

interval = prctile(distr, [50-interval/2 50+interval/2], dim);

end
