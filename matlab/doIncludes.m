src = fileparts(which('doIncludes'));

% addpath(fullfile(src, 'misc'))
addpath(fullfile(src, 'color'))
% addpath(fullfile(src, 'flow'))
% 
src_ext = fullfile(src, 'thirdparty');
% addpath(fullfile(src_ext, 'toolbox_diffc'));
% addpath(fullfile(src_ext, 'toolbox_diffc', 'toolbox'));
% addpath(fullfile(src_ext, 'colorspace'));
% addpath(fullfile(src_ext, 'matlabPyrTools'));
% addpath(fullfile(src_ext, 'motionMagnification'));
% addpath(fullfile(src_ext, 'motionMagnification', 'pyrToolsExt'));
% addpath(fullfile(src_ext, 'SIFTflow'));
% addpath(fullfile(src_ext, 'SIFTflow', 'mexDenseSIFT')); 
% addpath(fullfile(src_ext, 'SIFTflow', 'mexDiscreteFlow')); 
% addpath(fullfile(src_ext, 'opticalFlowMit'));
% addpath(fullfile(src_ext, 'opticalFlowMit', 'mex'));
% addpath(fullfile(src_ext, 'L0smoothing'));
% addpath(fullfile(src_ext, 'sintel'));
% addpath(fullfile(src_ext, 'FastAccurateBilateralFilter'));
% addpath(fullfile(src_ext, 'epicflow'));
% addpath(fullfile(src_ext, 'bbw_demo'));
% addpath(fullfile(src_ext, 'fullflow', 'Full_Flow_Source_Code'));
% addpath(fullfile(src_ext, 'fmgaussfit'));
% addpath(fullfile(src_ext, 'misc'));
% addpath(fullfile(src_ext, 'sRGB2CIEDeltaE'));
% addpath(fullfile(src_ext, 'SecretsOfFlow', 'flow_code_v2'));
% addpath(fullfile(src_ext, 'SecretsOfFlow', 'flow_code_v2', 'utils'));
% addpath(fullfile(src_ext, 'SecretsOfFlow', 'flow_code_v2', 'utils', 'downloaded'));
% addpath(fullfile(src_ext, 'HornSchunckFlow', 'hs'));
% addpath(fullfile(src_ext, 'HornSchunckFlow', 'hs', 'utils'));
% addpath(fullfile(src, 'misc', 'motion'))
% addpath(fullfile(src, 'misc', 'diffusion'))
addpath(fullfile(src_ext, 'InterPointDistanceMatrix', 'InterPointDistanceMatrix'))

% addpath(fullfile(src_ext, 'altmany-export_fig-26eb699'));