
small_val_patches = '/Users/shumash/Data/Color/splits/splits512/patch_splits/smval_target_patches/pimages';


do_graph = false;
default_options = {true, true, true, do_graph, 0.01, 25, []};
default_options = {true, true, true, do_graph, 0.00001, -1, [], 1};

exp_base = '/Users/shumash/Documents/Coding/Animation/animation/experiments/color2020/single_palette_fits/matlab_tuning2/sm_val/';



for quant=-1:1
    for kl=[0.1, 0.01,0.001,0.0001,0.00001,0.000001]
        t0 = tic;
        options = default_options;
        options{5} = kl;
        if quant == 0
            options{6} = 0;
            exp_name = 'noquant_';
        elseif quant == -1
            options{6} = -1;
            exp_name = 'histquant_';
        else
            options{6} = 25;
            exp_name = 'quant_';
        end
            
        exp_name = sprintf('%s%0.5f', exp_name, kl); 
        disp(strcat('----> EXP: ', exp_name));

        [~,~,~,losses] = fitSailsToImagesInDir(...
            small_val_patches, options, 16, strcat(exp_base, exp_name), [], [], false, -1);
        experiments(1).name = exp_name;
        experiments(1).losses = losses;
        experiments(1).time = toc(t0);
        toc(t0)
    end
end

