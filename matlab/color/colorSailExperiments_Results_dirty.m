
test_images = '/Users/shumash/Documents/Coding/Animation/animation/data/color/test_images/basic_onesail';
%exp_base = '/Users/shumash/Documents/Coding/Animation/animation/experiments/color2020/single_palette_fits/basic_results0/';
exp_base = '/Users/shumash/Documents/Coding/Animation/animation/experiments/color2020/single_palette_fits/experiments4/';
%exp_base = '/tmp/quick_matlab_test';


test_images = '/Users/shumash/Documents/Coding/Animation/animation/data/color/test_images/additions';
exp_base = '/Users/shumash/Documents/Coding/Animation/animation/experiments/color2020/single_palette_fits/additions1/';

test_images = '/Users/shumash/Documents/Coding/Animation/animation/experiments/color2020/single_palette_fits/MASKED/src_colors_for_fitting';
exp_base = '/Users/shumash/Documents/Coding/Animation/animation/experiments/color2020/single_palette_fits/MASKED/results0/';

test_images = '/Users/shumash/Documents/Coding/Animation/animation/experiments/color2020/single_palette_fits/COMPRESS/imgs/';
exp_base = '/Users/shumash/Documents/Coding/Animation/animation/experiments/color2020/single_palette_fits/COMPRESS/results1/';

test_images = '/Users/shumash/Documents/Coding/Animation/animation/experiments/color2020/single_palette_fits/TEASER/images/';
exp_base = '/Users/shumash/Documents/Coding/Animation/animation/experiments/color2020/single_palette_fits/TEASER/results/';

test_images = '/Users/shumash/Documents/Coding/Animation/animation/experiments/color2020/single_palette_fits/MASKED2/fit_input/';
exp_base = '/Users/shumash/Documents/Coding/Animation/animation/experiments/color2020/single_palette_fits/MASKED2/results/';

test_images= '/Users/shumash/Documents/Coding/Animation/animation/experiments/color2020/single_palette_fits/PRESO0/fit_input/';
exp_base = '/Users/shumash/Documents/Coding/Animation/animation/experiments/color2020/single_palette_fits/PRESO0/results/tst1/';

test_images= '/Users/shumash/Documents/Coding/Animation/animation/experiments/color2020/single_palette_fits/refinements1/fit_input/';
exp_base = '/Users/shumash/Documents/Coding/Animation/animation/experiments/color2020/single_palette_fits/refinements1/results/';

test_images= '/Users/shumash/Documents/Coding/Animation/animation/experiments/color2020/single_palette_fits/refinements0/fit_input2/';
exp_base = '/Users/shumash/Documents/Coding/Animation/animation/experiments/color2020/single_palette_fits/refinements0/results/new0/';

test_images = '/Users/shumash/Data/Images/Scribbles/paintlike'
exp_base = '/Users/shumash/Documents/Coding/Animation/animation/experiments/color2020/single_palette_fits/scribbles/test0'
%test_images = '/tmp/quant_exp/medusa_image/';
%test_images = '/tmp/quant_exp/images';
%test_images = '/tmp/quant_exp/cup_images';
%exp_base = '/tmp/quant_exp/results_no_quant/';  %  {true, true, true, do_graph, 0.01, -1, [], 1};
%exp_base = '/tmp/quant_exp/results_quant25/';  % {true, true, true, do_graph, 0.01, 25, [], 1}
%exp_base = '/tmp/quant_exp/results_quant25_sub10/'; % {true, true, true, do_graph, 0.01, 25, [], 10};
%exp_base = '/tmp/quant_exp/results_quant25_hybrid_blur/'; % {true, true, true, do_graph, 0.01, 25, [], -1}
%exp_base = '/tmp/quant_exp/results_hist_input/';  % {true, true, true, do_graph, 0.00, -1, [], -2};

do_graph = true;
default_options = {true, true, true, do_graph, 0.00001, -1, [], 1};
%default_options = {true, true, true, do_graph, 0.01, 25, [], 1};
%default_options = {true, true, true, do_graph, 0.01, 25, [], -1};
img_width = 512;

% subexp(1).name = 'noquant_nokl';
% subexp(1).options = {true, true, true, false, 0.00, 0, [], 1};
% 
% subexp(2).name = 'basicquant_nokl';
% subexp(2).options = {true, true, true, false, 0.00, 25, [], 1};
% 
subexp(3).name = 'histquant_nokl';
subexp(3).options = {true, true, true, do_graph, 0.00, -1, [], 1};

subexp(4).name = 'noquant_kl0.0001';
subexp(4).options = {true, true, true, do_graph, 0.0001, 0, [], 1};

% subexp(5).name = 'basicquant_kl0.0001';
% subexp(5).options = {true, true, true, false, 0.0001, 25, [], 1};
% 
% subexp(6).name = 'histquant_kl0.0001';
% subexp(6).options = {true, true, true, false, 0.0001, -1, [], 1};


for i=4 %3:4 %3:4
      disp(strcat('Experiment: ', subexp(i).name));
      subexp_base = strcat(exp_base, subexp(i).name, '/');
    exp_name = 'palettes_16';
    optimize_n = false;
    opts = subexp(i).options;
    [~,~,~,losses] = fitSailsToImagesInDir(...
        test_images, opts, 16, strcat(subexp_base, exp_name), [], [], optimize_n, img_width);

    % 'mountains'
    
%     exp_name = 'palettes_deeptune16';
%     optimize_n = false;
%     opts = subexp(i).options;
%     [~,~,~,losses] = fitSailsToImagesInDir(...
%         test_images, opts, 16, strcat(subexp_base, exp_name), ...
%         strcat(subexp_base, 'palettes_deep'), [], optimize_n, img_width);
    
%     exp_name = 'palettes';
%     optimize_n = true;
%     opts = subexp(i).options;
%     opts{4} = false;
%     [~,~,~,losses] = fitSailsToImagesInDir(...
%         test_images, opts, 16, strcat(subexp_base, exp_name), [], [], optimize_n, img_width);
    
%     exp_name = 'palettes_deeptune';
%     optimize_n = true;
%     opts = subexp(i).options;
%     opts{4} = false;
%     [~,~,~,losses] = fitSailsToImagesInDir(...
%         test_images, opts, 16, strcat(subexp_base, exp_name), ...
%         strcat(subexp_base, 'palettes_deep'), [], optimize_n, img_width);
    
    
end