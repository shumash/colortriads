function losses = colorSailExperiments_Results(...
    test_images, exp_base, do_graph, n_subdiv, opts, img_width, optimize_n, exp_name)
    % Run color sail experiments with specified test images and output directory
    % Args:
    %   test_images: Path to directory containing test images
    %   exp_base: Base path for experiment outputs
    %   do_graph: (optional) Whether to show graphs, default true
    %   n_subdiv: (optional) Number of subdivisions, default 16
    %   opts: (optional) Cell array of options with following elements:
    %       {1} optimize_wind: boolean
    %       {2} optimize_colors: boolean
    %       {3} optimize_pressure_pt: boolean
    %       {4} do_graph: boolean (ignored, use do_graph arg instead)
    %       {5} kl_weight: double
    %       {6} nquant_bins: integer
    %       {7} orig_image: array or []
    %       {8} nquant_subsets: integer
    %   img_width: (optional) Width to resize images to, default 512
    %   optimize_n: (optional) Whether to optimize number of subdivisions, default false
    %   exp_name: (optional) Base name for experiment directory, default 'palettes'

    % Set default parameters if not provided
    if nargin < 3
        do_graph = true;
    end
    if nargin < 4
        n_subdiv = 16;
    end
    if nargin < 5
        opts = {true, true, true, do_graph, 0.00001, -1, [], 1};
    end
    if nargin < 6
        img_width = 512;
    end
    if nargin < 7
        optimize_n = false;
    end
    if nargin < 8
        exp_name = 'palettes';
    end

    % Generate experiment name and directory
    exp_dir = strcat(exp_base, exp_name);

    fprintf('Running Color Triad (Color Sail) fitting on an image directory, with settings:\n');
    fprintf('  Input directory: %s\n', test_images);
    fprintf('  Output directory: %s\n', exp_dir);
    fprintf('  Image width: %d\n', img_width);
    fprintf('  Max Subdivisions: %d\n', n_subdiv);
    fprintf('  Optimize subdivisions: %d\n', optimize_n);
    fprintf('  Options:\n');
    fprintf('    optimize_wind: %d\n', opts{1});
    fprintf('    optimize_colors: %d\n', opts{2});
    fprintf('    optimize_pressure_pt: %d\n', opts{3});
    fprintf('    kl_weight: %f\n', opts{5});
    fprintf('    nquant_bins: %d\n', opts{6});
    fprintf('    nquant_subsets: %d\n', opts{8});
    fprintf('\n');
    
    [~,~,~,losses] = fitSailsToImagesInDir(...
        test_images, opts, n_subdiv, exp_dir, [], [], optimize_n, img_width);
end