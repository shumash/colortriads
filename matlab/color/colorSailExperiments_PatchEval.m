
%input_patches = '/tmp/smpatches/';
%exp_base = '/tmp/smpatches_testrun/';

input_patches = '/Users/shumash/Data/Color/splits/splits512/patch_splits/test_target_patches/pimages_3K/';
exp_base = '/Users/shumash/Documents/Coding/Animation/animation/experiments/color2020/single_palette_fits/K3_testpatch_eval/';
% ./learn/tf/color/run_fitting_experiment.sh /Users/shumash/Data/Color/splits/splits512/patch_splits/test_target_patches/pimages_3K  /Users/shumash/Documents/Coding/Animation/animation/experiments/color2020/single_palette_fits/K3_testpatch_eval/


% options = {true, true, true, false, 0.0001, 0, [], 1};
% [~,~,~,losses] = fitSailsToImagesInDir(...
%             input_patches, options, 16, strcat(exp_base, 'palettes_16'), [], [], false, -1);
        
exp_name = 'palettes_deeptune16_FAKE';
[~,~,~,losses] = fitSailsToImagesInDir(...
        input_patches, options, 16, strcat(exp_base, exp_name), ...
        strcat(exp_base, 'palettes_deep'), [], false, -1);
    