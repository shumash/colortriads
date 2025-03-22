function [colors, wind, sail_colors, meta] = colorSailOptimize(input_colors, wind, orig_target_colors, opts, nsubdiv)
% options are a cell array containing:
% {optimize_wind, optimize_colors, optimize_pressure_pt, graph}
% losses: total_loss l2_loss kl_loss percent_loss

% Get closest colors
% Optimize vertex colors and middle control point
% Recompute closest colors
% Repeat

optimize_wind = opts{1};
optimize_colors = opts{2};
optimize_pressure_pt = opts{3};
do_graph = opts{4};
kl_weight = opts{5};
nquant_bins = opts{6};
orig_image = opts{7};
nquant_subsets = opts{8};

wind = reshape(wind, [1,3]);

target_color_weights = [];
target_hist = [];
summary = sprintf('nsub=%d', nsubdiv);
if nquant_bins == 0
    target_colors = orig_target_colors;
    summary = strcat(summary, 'noquant');
elseif nquant_bins >= 1
    target_colors = binQuantizeColorSamples(orig_target_colors, nquant_bins, true, nquant_subsets);
    if nquant_subsets == -1 && ~isempty(orig_image)
       additional_samples = sampleImage(orig_image, floor(size(target_colors, 1) / 2), false);
       target_colors = [target_colors; additional_samples(:, 1:3)];
    end
    summary = strcat(summary, sprintf(' quant:%d,patches%d', nquant_bins, nquant_subsets));
elseif nquant_bins == -1
    if isempty(orig_image)
        error('Must specify original image if nquant_bins=-1')
    end
    [target_hist, exact_hist_centers] = computePatch3DHist(orig_image, 512, 32, 10, true);
    [bin_indices, bin_colors] = getHistogramCellCenters(10, exact_hist_centers);
    target_color_weights = target_hist(sub2ind(size(target_hist), bin_indices(:, 1), bin_indices(:, 2), bin_indices(:, 3)));
    nz = target_color_weights > 0;
    target_color_weights = target_color_weights(nz);
    target_colors = bin_colors(nz, :);
    
    additional_samples = sampleImage(orig_image, size(target_colors, 1), false);
    target_colors = [target_colors; additional_samples(:, 1:3)];
    target_color_weights = [target_color_weights; ones([size(additional_samples, 1), 1]) / size(additional_samples, 1)];
    
    summary = strcat(summary, sprintf(' histinput'));
end

if isempty(orig_image)
    target_hist = compute3DHistogram(target_colors, 10);
    summary = strcat(summary, sprintf(' KL:%f * H', kl_weight));
else
    if isempty(target_hist)
        [target_hist, ~] = computePatch3DHist(orig_image, 512, 32, 10, true);
    end
    summary = strcat(summary, sprintf(' KL:%f * Hpatch', kl_weight));
end

Ntargets = size(target_colors, 1);

if isempty(target_color_weights)
    target_color_weights = ones([Ntargets, 1]) / Ntargets;
end

if do_graph
    f = figure;
    sz = target_color_weights .* (220 / max(target_color_weights)) + 5;
    scatter3(target_colors(:, 1), target_colors(:, 2), target_colors(:, 3), sz, target_colors, 'filled');
end

Nvars = 0;
constant_vals = [];
x0 = [];
if optimize_colors
    Nvars = Nvars + 9;
    x0 = input_colors(:)';
else
    constant_vals = input_colors(:)';
end
if optimize_wind
    Nvars = Nvars + 1;
    x0 = [x0, wind(1)];
else
    constant_vals = [constant_vals, wind(1)];
end
if optimize_pressure_pt
    Nvars = Nvars + 3;
    wtmp = [wind(2:3), 1 - sum(wind(2:3), 'all')];
    x0 = [x0, log(wtmp)];
else
    constant_vals = [constant_vals, wind(2:3)];
end

summary = strcat(summary, sprintf(' Nvars=%d', Nvars));

[bary, cw, bernst, cbernst] = bezierTriangleWeights(nsubdiv, 3);


lb = zeros(1, Nvars);
ub = ones(1, Nvars);

if optimize_wind  % wind ranges between -1 and 1
    if Nvars > 3
        lb(10) = -1.0;
    else
        lb(1) = -1.0;
    end
end

[loss, sail_colors, ~, l2_loss, kl_loss] = lossFunctionUtil(...
    target_colors, target_color_weights, orig_target_colors, target_hist, kl_weight, bernst, constant_vals, x0);
err = getPercentError(rgb2lab(orig_target_colors), rgb2lab(sail_colors), 10);
meta(1).start_losses = full([loss, l2_loss, kl_loss, err]);
start_loss = loss;

if do_graph
    hold on;
    scatter3(sail_colors(:, 1), sail_colors(:, 2), sail_colors(:, 3), 20, [0.8, 0.8, 0.8], 'filled');
    title(summary);
end

%foptions = optimoptions('fmincon','Algorithm','sqp');
[x, ~, exitflag, optinfo] = fmincon(@(x)lossFunction(...
    target_colors, target_color_weights, orig_target_colors, target_hist, kl_weight, bernst, constant_vals, x), ...
    x0, [], [], [], [], lb, ub); %, [], foptions);
meta(1).exitflag = exitflag;
meta(1).optinfo = optinfo;


[loss, sail_colors, ~, l2_loss, kl_loss] = lossFunctionUtil(...
    target_colors, target_color_weights, orig_target_colors, target_hist, kl_weight, bernst, constant_vals, x);
final_loss = loss;

[colors, wind] = extractVals(x, constant_vals);
if do_graph
    hold on;
    bezierTriangleClean(colors, wind, nsubdiv);
    figure(f);
    title(summary);
    %scatter3(sail_colors(:, 1), sail_colors(:, 2), sail_colors(:, 3), 20, [0, 0, 0], 'filled');
end

err = getPercentError(rgb2lab(orig_target_colors), rgb2lab(sail_colors), 10);
meta(1).loss_labels = {'loss', 'l2_loss', 'kl_loss', 'percent_loss'};
meta(1).losses = full([loss, l2_loss, kl_loss, err]);

disp(sprintf('Loss: %0.4f --> %0.4f', start_loss, final_loss));
end

function loss = lossFunction(target_colors, target_color_weights, orig_target_colors, ...
    target_hist, kl_weight, bernst, constant_vals, x)

[loss, ~, ~, ~, ~] = lossFunctionUtil(target_colors, target_color_weights, orig_target_colors, ...
    target_hist, kl_weight, bernst, constant_vals, x);

end

% 9, 10, 12, 1, 2, 3
function [verts, wind] = extractVals(x, constant_vals)
cv_idx = 1;
x_idx = 1;
if numel(x) >= 9
    verts = x(1:9);
    x_idx = x_idx + 9;
else
    verts = constant_vals(1:9);
    cv_idx = cv_idx + 9;
end

if numel(x) == 1  || numel(x) == 10 % only optimize wind value
    wind = [x(x_idx), constant_vals(cv_idx:cv_idx+1)];
elseif numel(x) == 3 || numel(x) == 12 % only optimize pressure
    w_tmp = exp(x(x_idx:x_idx+2));
    w_tmp = w_tmp / sum(w_tmp, 'all');  % softmax to ensure sum = 1
    wind = [constant_vals(cv_idx), w_tmp(1:2)];
elseif numel(x) == 4 || numel(x) == 13 % optimize wind and pressure
    w_tmp = exp(x(x_idx+1:x_idx+3));
    w_tmp = w_tmp / sum(w_tmp, 'all');  % softmax to ensure sum = 1
    wind = [x(x_idx), w_tmp(1:2)];
end

verts = reshape(verts, [3, 3]);
end

function [loss, sail_colors, approx_colors, l2_loss, kl_loss] = lossFunctionUtil(...
    target_colors, target_color_weights, orig_target_colors, target_hist, kl_weight, bernst, constant_vals, x)

[verts, wind] = extractVals(x, constant_vals);

inflation = wind(1);
u = wind(2);
v = wind(3);
sail_colors = getSailColorsInternal(verts, inflation, u, v, bernst);

D = ipdm(target_colors, sail_colors, 'Subset','nearest');
l2_loss = full(sum(sum(D .* D, 2) .* target_color_weights, 'all'));

%loss = getPercentError(rgb2lab(target_colors), rgb2lab(sail_colors), 10);  % HACK
kl_loss = getKLError(target_hist, sail_colors);

loss = l2_loss + kl_weight * kl_loss;

%[r, c] = find(D);
approx_colors = [];  % TODO

end

function colors = getSailColorsInternal(verts, inflation, u, v, bernst)
pts = getSailControlPoints(verts, inflation, u, v);
pts = reshape(pts, [1, size(pts, 1), size(pts, 2)]);
colors = squeeze(sum(bernst .* pts, 2));
end

function err = getPercentError(target_colors_lab, sail_colors_lab, delta)
D = ipdm(target_colors_lab, sail_colors_lab, 'Subset','nearest');
D = sum(D, 2);
err = sum(D > delta) / numel(D);  
end

function kl = getKLError(target_hist, sail_colors)
nbins = 10;
hs = computeSoft3DHistogram(sail_colors, nbins);
kl = computeHistKLDivergence(hs, target_hist);
end

function kl = getKLErrorRaw(target_colors, sail_colors)
nbins = 10;
hs = computeSoft3DHistogram(sail_colors, nbins);
ht = compute3DHistogram(target_colors, nbins);
kl = computeHistKLDivergence(hs, ht);
end
