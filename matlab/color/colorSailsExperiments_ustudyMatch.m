% 
% ustudy_base = '';
% output_base = '';
% 
% img_id = 'blue';
% % /Users/shumash/Downloads/ustudy_blue_s10_palette.txt
% % /Users/shumash/Downloads/ustudy_blue_s1_palette.txt
% % /Users/shumash/Downloads/ustudy_blue_s4_palette.txt
% % /Users/shumash/Downloads/ustudy_blue_s7_palette.txt
% uid='s10';
% uid='s1';
% uid='s4';
% uid='s7';
% 
% img_id = 'green';
% % /Users/shumash/Downloads/ustudy_green_s11_palette.txt
% % /Users/shumash/Downloads/ustudy_green_s2_palette.txt
% % /Users/shumash/Downloads/ustudy_green_s5_palette.txt
% % /Users/shumash/Downloads/ustudy_green_s8_palette.txt
% uid='s11';
% uid='s2';
% uid='s5';
% uid='s8';
% 
% img_id = 'red';
% %pfile = '/Users/shumash/Downloads/ustudy_red_s3_palette.txt';
% %pfile = '/Users/shumash/Downloads/ustudy_red_s6_palette.txt';
% %pfile = '/Users/shumash/Downloads/ustudy_red_s9_palette.txt';
% uid='s3';
% uid='s6';
% uid='s9';

default_options = {true, true, true, false, 0.00001, -1, [], 1};

in_palette_dir = '/Users/shumash/Documents/Coding/Animation/animation/experiments/color2020/single_palette_fits/USTUDY_match/raw_results';
src_images = '/Users/shumash/Documents/Coding/Animation/animation/experiments/color2020/single_palette_fits/USTUDY_match/src_images/';
out_palette_dir = '/Users/shumash/Documents/Coding/Animation/animation/experiments/color2020/single_palette_fits/USTUDY_match/fitted_palettes/';

exp_name = 'palettes_16';
optimize_n = true;
img_width = 256;
[~,~,~,losses] = fitSailsToImagesInDir(...
        src_images, default_options, 16, strcat(out_palette_dir, exp_name), in_palette_dir, [], optimize_n, img_width);

        
        