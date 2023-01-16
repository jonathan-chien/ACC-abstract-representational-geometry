function [binStarts,binCenters,nBins] = get_bin_timepoints(nvp)
% Takes as inputs a series of name-value pairs describing the analysis
% window (beginning and end timepoints), bin width, sliding nature (true or
% false), and step size, then vectors of timepoints corresponding to bin
% starts of bin centers.
%
% PARAMETERS
% ----------
% Name-Value Pairs (nvp)
%   'window'      -- [1 x 2] array containing the timepoint of the
%                    beginning and the end of the time window over which
%                    timepoints are desired. Values are with respect to the
%                    defining event of each epoch (i.e. stim on for
%                    stimulus epoch, response for response epoch, feedback
%                    for feedback epoch) and should be set precisely, such
%                    as [101 600] if looking from 101st ms to 600th ms
%                    (default).
%   'binWidth'    -- Scalar value that is width in ms of a single bin. Must
%                    be even. Default = 150.
%   'sliding'     -- Logical true (default) or false. Specify whether or
%                    not bins slide with overlap (if true) or are discrete
%                    (if false). 
%   'step'        -- Scalar value specifying how far along (in ms) bins
%                    slide forward from one bin to the next when bins
%                    overlap. Not used if 'sliding' set to false. Default =
%                    25.
%   'wrt'         -- String value, either 'window' or 'event'. Name is the
%                    abbreviation for "with respect to" and indicates
%                    whether vector of binStarts (and hence binCenters
%                    which is calculated by offsetting binStarts by a
%                    scalar amount) should feature timepoints with respect 
%                    to the window bounds (and thus beginning at 1, useful
%                    for indexing within a raster, etc.) or with respect to
%                    the epoch-defining event (e.g., beginning at -999 for
%                    a window = [-999 1500], which may be useful when
%                    converting from bin indices to timepoints and vice
%                    versa). Default is 'window'.
%
% RETURNS
% -------
% binStarts  -- 1 x nBins of timepoints corresponding to the start (1st
%               timepoint) of time bins in the specified analysis window
%               ('window').
% binCenters -- 1 x nBins of timepoints corresponding to the center of time
%               bins in the specified analysis window ('window').
% nBins      -- Scalar value equal to number of bins (length of binStarts
%               binCenters).
%
% Author: Jonathan Chien Version 1.0 6/27/21. Last edit: 2/7/22.
%
% Version history:
%   -- This code originally appeared in constructBinnedPopRespTrials.m from
%      the Machine Learning (BMI3002) project (spring 2021). It since then
%      has appeared in many other functions as well.
%   -- Change to method used to calculate window width 10/13/21
%      (simplified).
%   -- (2/7/22) Removed assertion that binWidth be even (now checks parity
%      and calculates binCenters in different manners for even vs odd
%      binWidth.

arguments
    nvp.window (1,2) = [-199 1500]
    nvp.binWidth (1,1) {mustBeInteger} = 150
    nvp.sliding = true
    nvp.step = 25
    nvp.wrt string = 'window' 
end

% Check parity of binWidth.
if mod(nvp.binWidth, 2) == 0, parity = 'even'; else, parity = 'odd'; end

% Define analysis window (timepoints are wrt to the defining event of each
% epoch (i.e. stim on for stimulus epoch, response for response epoch,
% feedback for feedback epoch). Window bounds ('window' nvp) should be set
% precisely, such as [101 600] if looking from 101st ms to 600th ms.
windowWidth = nvp.window(2) - nvp.window(1) + 1;

% Calculate timepoints corresponding to start and center of bins.
if nvp.sliding
    % Check to ensure valid parameters.
    assert(mod((windowWidth-nvp.binWidth),nvp.step)==0, ...
           ['Step size, bin width, or both are invalid. Window width ' ...
            'minus bin width must be evenly divisible by step size.'])
    
    % Set first timepoint as 1, or as window edge with respect to
    % epoch-defining event?
    switch nvp.wrt
        case 'window'
            binStarts = 1 : nvp.step : windowWidth - nvp.binWidth + 1;
        case 'event'
            binStarts = nvp.window(1) : nvp.step : nvp.window(2) - nvp.binWidth + 1;
    end
    
    % Offset binStarts to get binCenters.
    if strcmp(parity, 'even')
        binCenters = binStarts + (nvp.binWidth/2 - 1);
    elseif strcmp(parity, 'odd')
        binCenters = binStarts + ((nvp.binWidth - 1) / 2);
    end
    
else
    % Check to ensure valid parameters.
    assert(mod(windowWidth, nvp.binWidth) == 0, ...
           ['Discrete bins requested, but specified window width ' ...
            'cannot be evenly divided by specified bin width.'])
    
    % Set first timepoint as 1, or as window edge with respect to
    % epoch-defining event?
    switch nvp.wrt
        case 'window'
            binStarts = 1 : nvp.binWidth : windowWidth;
        case 'event'
            binStarts = nvp.window(1) : nvp.binWidth : nvp.window(2) - nvp.binWidth + 1;
    end
    
    % Offset binStarts to get binCenters.
    if strcmp(parity, 'even')
        binCenters = binStarts + (nvp.binWidth/2 - 1);
    elseif strcmp(parity, 'odd')
        binCenters = binStarts + ((nvp.binWidth - 1) / 2);
    end
end

% Determine number of bins.
nBins = length(binStarts);

end