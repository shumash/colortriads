function [colors, wind, meta, loss_summary] = fitSailsToImagesInDir(dir_name, opts, nsubdiv, out_dir, ...
    prefit_palette_dir, match_name, optimize_n, imwidth)
% Same opts as colorsailoptimize
% total_loss l2_loss kl_loss percent_loss

do_graph = opts{4};

files = [dir(strcat(dir_name, '/', '*.jpg')); dir(strcat(dir_name, '/', '*.png'))];
disp(out_dir)
mkdir(out_dir)

res_file = [];
init_file = [];
loss_summary = [];

for file = files'
    start_t = tic;
    [~, name, ~] = fileparts(file.name);
    if ~isempty(match_name) && ~strcmp(name, match_name)
        disp(sprintf('Skipping %s', file.name));
        continue
    end
    
    in_img = imread(fullfile(dir_name, file.name));
    if imwidth > 0
        in_img = imresize(in_img, [imwidth, imwidth]);
        in_img = imgaussfilt(in_img); 
    end
    opts{7} = in_img;
    S = sampleImage(in_img, min(10000, size(in_img, 1) * size(in_img, 2)), false);
    U = S(:, 1:3);  % Quantization happens in the fitting function, if specified
    
    if ~isempty(prefit_palette_dir)
       start_palette_file = strcat(prefit_palette_dir, '/', name, '.palette.txt');
       if ~isfile(start_palette_file)
          error(strcat('Cannot find file: %s', start_palette_file)) 
       end
       disp(strcat('Starting from palette: ', start_palette_file));
       [start_colors, start_wind, ~] = colorSailFromFile(start_palette_file);
       disp('Read starter palette');
       [colors, wind, ~, meta] = colorSailOptimize(start_colors, start_wind, U, opts, nsubdiv);
    else
        [colors, wind, ~, meta] = colorSailOptimizeFromScratch(U, opts, nsubdiv);
    end
    
    if optimize_n
       [colors, wind, opt_nsubdiv, opt_n_meta] = colorSailOptimizeNsubdiv(colors, wind, U, opts, nsubdiv); 
       meta = opt_n_meta(opt_nsubdiv);
    else
        opt_nsubdiv = nsubdiv;
    end
    
%     if do_graph
%         title(name);
%     end
    
    palette_fname = strcat(out_dir, '/', name, '.palette.txt');
    colorSailToFile(palette_fname, colors, wind, opt_nsubdiv);
    disp(sprintf('Wrote palette: %s', palette_fname));
    
    if isempty(res_file)
        res_file = fopen(strcat(out_dir, '/', 'losses.txt'), 'w');
        fprintf(res_file, 'name %s %s\n', strjoin(meta(1).loss_labels), 'total_time');
        init_file = fopen(strcat(out_dir, '/', 'init_losses.txt'), 'w');
        fprintf(init_file, 'name %s\n', strjoin(meta(1).loss_labels));
    end
    
    total_time = toc(start_t);
    fprintf(res_file, '%s %0.4f %0.4f %0.4f %0.4f %0.4f\n', file.name, ...
        meta(1).losses(1), meta(1).losses(2), meta(1).losses(3), meta(1).losses(4), total_time);
    fprintf(init_file, '%s %0.4f %0.4f %0.4f %0.4f\n', file.name, ...
        meta(1).start_losses(1), meta(1).start_losses(2), meta(1).start_losses(3), meta(1).start_losses(4));
    loss_summary(size(loss_summary, 1) + 1, :) = meta(1).losses;
end

fclose(res_file);
fclose(init_file);
end