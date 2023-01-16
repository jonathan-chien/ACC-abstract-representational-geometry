function ps = calc_ps(T,condLabels,nv)
% Calculates "parallelsim score" (PS) as described in Bernardi et al "The
% Geometry of Abstraction in the Hippocampus and Prefrontal Cortex," Cell,
% 2020. Returns struct containing the parallelism score for the given data,
% as well as associated metrics of statistical significance.
%
% PARAMETERS
% ----------
% T          : t x n matrix, where t = number of trials and n = number of
%              neurons. The i_th row j_th column element is the firing
%              rate of the j_th neuron on the i_th single trial. All
%              trials (rows of T) belong to one of q conditions. Note that
%              each of the p conditions is expected to be represented t /
%              q times among the rows of T (which may be achived via
%              oversampling etc). 
% condLabels : Vector of length t, where the i_th element is a
%              positive integer condition label corresponding to the i_th
%              row (single trial) of T. The number of conditions is
%              calculated by the function as the number of unique elements
%              in the condLabels vector.
% Name-Value Pairs (nv)
%   'condNames'      : ([] (default) | 1 x n_conds cell array), where
%                      each cell contains the name/description of a
%                      condition. Default is an empty array, in which case
%                      the ps struct will be returned without the
%                      dichotomy_conds field (see RETURNS).
%   'dropInd'        : ([] (default) | 1 x m vector), where each element
%                      is the index of a neuron within the population that
%                      we wish to drop (i.e. all its entries across trials
%                      are deleted, so that the length of the second
%                      dimension of T overall decreases by m) before
%                      calculating PS.
%   'pval'           : ('two-tailed' (default) | 'left-tailed' |
%                      'right-tailed' | false). If false, null
%                      distributions for PS will not be computed. If one of
%                      the above string values, p values (and confidence
%                      intervals/zscores) will be computed accordingly from
%                      null distributions for PS (see 'nullMethod' below).
%   'nullInt'        : (95 (default) | positive integer in [0, 100]).
%                      Specify the interval size (as a percentage) to be
%                      calculated around the mean of the null
%                      distributions.
%   'nNull'          : (1000 (default) | positive integer). Specify
%                      number of synthetic/null datasets to generate in
%                      order to assess statistical significance of PS.
%   'nullMethod'     : ('permutation' (default) | 'geometric'). Specify
%                      the method to be used to generate a null dataset. If
%                      'permutation', entries of each column vector
%                      (corresponding to trials for one neuron) of the
%                      original input T are shuffled independently, and the
%                      process of calculating CCGP is carried out as on the
%                      empirical data. This destroys cluster structure to a
%                      certain extent (though it may fail to do so in
%                      certain cases, e.g. if the distribution of all
%                      matrix elements is multimodal). If 'geometric',
%                      n_conds n-vectors (n = number of neurons) are
%                      sampled from a n-dim MVN distribution with 0 mean
%                      and covarance matrix given by n x n identity matrix.
%                      These vectors are considered random cluster
%                      centroids and point clouds corresponding to each
%                      condition in the empirical data are rotated and then
%                      moved to the new centroids in n-space; see
%                      construct_random_geom for more details.
%   'returnNullDist' : (1 | 0 (default)). Specify whether or not to
%                      return the null distribution for the parallelism
%                      score (for each dichotomy) in the ps struct (see
%                      RETURNS below). Ignored if 'pval' = false, since no
%                      null distributions are generated in that case.
% 
% RETURNS
% -------
% ps             -- 1 x 1 struct with the following fields:
%   .ps             -- nDichotomies x 1 vector, where the i_th element is
%                      the parallelism score for the i_th dichotomy in the
%                      input data.
%   .p              -- nDichotomies x 1 vector, where the i_th element is
%                      the p value attached to the parallelism score for
%                      the i_th dichotomy.
%   .nullInt        -- nDichotomies x 2 matrix, where each row corresponds
%                      to a dichotomy. For each row/dichotomy, the first
%                      and second column elements are the upper and lower
%                      bounds, respectively, of the specified interval
%                      around the null distribution mean; the size of this
%                      interval was specified through the 'nullInt' name
%                      value pair (default size is 95).
%   .obsStdev       -- nDichotomies x 1 vector, where each element is the
%                      number of standard deviations from the mean of the
%                      i_th dichotomy's null distribution that lies the
%                      observed parallelism score on the i_th dichotomy.
%   .nullDist       -- nDichotomies x nNull array (see 'nNull' under
%                      Name-Value Pair options above) whose i_th j_th
%                      element is the j_th draw from the parallelism score
%                      null distribution for the i_th dichotomy.
%   .dichotomyConds -- An optionally returned field, dichtomyConds is an
%                      nDichotomies x nConds cell array where each row
%                      corresponds to a dichotomy. For each row
%                      (dichotomy), the first 1:nConds/2 cells contain the
%                      labels of the conditions on one side of the
%                      dichotomy and the last nConds/2+1:end cells contain
%                      the labels of the conditions on the other side of
%                      the dichotomy. If condNames is empty (as it is by
%                      default), this field will be absent from ps.
%
% Author: Jonathan Chien 7/23/21. Last edit: 2/4/22.

arguments
    T
    condLabels
    nv.condNames = []
    nv.dropIdc = []
    nv.nNull = 1000
    nv.nullInt = 95
    nv.nullMethod = 'geometric'
    nv.pval = 'two-tailed'
    nv.returnNullDist = false
end


%% Preprocess inputs

% Option to drop neurons if desired, then obtain number of neurons.
T(:,nv.dropIdc) = [];
nNeurons = size(T, 2);

% Determine number of conditions and set combination parameter m, where we
% want to partition m*n objects into n unique groups, with n = 2 (hence a
% "dichotomy").
nConds = length(unique(condLabels));
m = nConds / 2;

% Get dichotomy indices and labels (if provided). Store labels as field of
% ps at end to preserve order of fields.
[dichotomies,dichotomyConds] = create_dichotomies(nConds, nv.condNames);
nDichotomies = size(dichotomies, 1);

% Calculate mean of each condition (centroid of each empirical condition
% cluster).
CxN = calc_centroids(T, nConds);


%% Calculate Parallelism Score (PS)

% Preallocate.
parScore = NaN(nDichotomies, 1);

% Calculate parallelism score (PS) for all dichotomies.
parfor iDichot = 1:nDichotomies
    
    % Obtain indices of condtitions on either side of current dichotomy.
    side1 = dichotomies(iDichot,1:m);
    side2 = dichotomies(iDichot,m+1:end);
    
    % Take max of the mean cosine similarities from each permutation as the
    % PS for the current dichotomy.
    parScore(iDichot) = max(linking_vecs(CxN, side1, side2));
end

ps.ps = parScore;


%% Compute null distribution for PS

if nv.pval
    
% Preallocate.
nullParScore = NaN(nDichotomies, nv.nNull);

% Generate distribution of PS scores for each dichotomy by repeating above
% process 'nNull' times.
parfor iNull = 1:nv.nNull    
    
    % Construct null model via permutation.
    if strcmp(nv.nullMethod, 'permutation')
        nullTxN = NaN(size(T));
        nTrials = size(T, 1);
        
        for iNeuron = 1:nNeurons
            nullTxN(:,iNeuron) = T(randperm(nTrials),iNeuron);
        end 
        
        nullCxN = calc_centroids(nullTxN, nConds);
       
    % Construct null model via geometric model.
    elseif strcmp(nv.nullMethod, 'geometric')
        nullCxN = construct_random_geom(T, nConds, 'addTrials', false);
    end
    
    % Calculate parallelism score (PS) for each dichotomy.
    currParScore = NaN(nDichotomies, 1);
    for iDichot = 1:nDichotomies
    
        % Obtain indices of condtitions on either side of current dichotomy.
        side1 = dichotomies(iDichot,1:m);
        side2 = dichotomies(iDichot,m+1:end);

        % Take max of the mean cosine similarities from each permutation as
        % the PS for the current dichotomy.
        currParScore(iDichot) = max(linking_vecs(nullCxN, side1, side2));
    end
    
    % Store PS of all dichotomies from current null run.
    nullParScore(:,iNull) = currParScore;
end

% Option to return null distribution of PS for each dichotomy.
if nv.returnNullDist, ps.nullDist = nullParScore; end


%% Compute p values and null intervals

% Not foolproof if somehow n_dichotomies = n_null but generally should
% guard against issues due to changing variable shapes.
NULLDIM = 2;
assert(size(nullParScore, NULLDIM) == nv.nNull);

% Calculate p value.
ps.p = tail_prob(parScore, nullParScore, NULLDIM, 'type', nv.pval);

% Caclulate interval around mean of null distribution. 
ps.nullInt = interval_bounds(nullParScore, nv.nullInt, NULLDIM);

% Calculate number of st.dev.'s away from null mean that empirical value
% lies.
ps.zscore = null_zscore(parScore, nullParScore, NULLDIM);

end

% Add names of conditions in dichotomies if condition labels provided.
if ~isempty(dichotomyConds), ps.dichotomyConds = dichotomyConds; end

end


% --------------------------------------------------
function meanCosineSim = linking_vecs(CxN, side1, side2)
% For a given dichotomy (each side's conditions indices are in side1 and
% side2), compute the vector differences (linking or coding vectors)
% between condition centroids from side1 and side2 across all possible ways
% of matching condition centroids from the two sides of the dichotomy.
%
% PARAMETERS
% ----------
% CxN   -- nConditions x nNeurons matrix of firing rates.
% side1 -- Vector of length nConditions/2 containing condition indices from
%          side 1 of the current dichotomy.
% side2 -- Vector of length nConditions/2 containing condition indices from
%          side 2 of the current dichotomy.
%
% RETURNS
% -------
% meanCosineSim - nPermutations x 1 vector whose i_th element is the
%                 mean cosine similarity across all pairs of linking vectors
%                 for one way (permutation of side2) of matching up
%                 condition centroids from the two sides of the dichotomy.
%
% Jonathan Chien. 8/23/21.

% Number of neurons and conditions.
[nConds, nNeurons] = size(CxN); 
m = nConds/2;

% Obtain all permutations of side2. Also prepare indices of mchoose2
% used to index pairs of linking vectors, calculate number of pairs, and
% preallocate.
side2Perms = perms(side2);
nPerms = size(side2Perms, 1);
pairIdc = nchoosek(1:m, 2); % each row is a pair of linking vecs
nPairs = size(pairIdc, 1);
linkingVecPairs = NaN(2, nPairs, nNeurons);
meanCosineSim = NaN(nPerms, 1);

% Define linking vectors (one to one match) between side 1 and each
% permutation of side 2; these are vector differences. E.g., there are
% four linking vectors in the case of 8 conditions.
for iPerm = 1:nPerms
    
    % Calculate vector difference between the i_th condition vector
    % (averaged over all trials of that condition) in side1 and the
    % i_th condition vector (also averaged over trials) in the current
    % permutation of side2 for all i from 1 to m.
    linkingVecs = CxN(side1,:) - CxN(side2Perms(iPerm,:),:);
    
    % Determine all unique ways to pair up the linking vectors for the
    % current permutation, then calculate cosine similarity of all
    % pairs.
    linkingVecPairs(1,:,:) = linkingVecs(pairIdc(:,1),:);
    linkingVecPairs(2,:,:) = linkingVecs(pairIdc(:,2),:);
    normedlinkingVecPairs = linkingVecPairs ./ vecnorm(linkingVecPairs, 2, 3);
    cosineSims = sum(normedlinkingVecPairs(1,:,:) ...
                     .* normedlinkingVecPairs(2,:,:), ...
                     3);
    meanCosineSim(iPerm) = mean(cosineSims);
end

end
