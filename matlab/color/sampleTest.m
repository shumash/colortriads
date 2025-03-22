function rgb_samples = sampleTest(inrgb, nsamples, sample_space)

if strcmp(sample_space, 'LAB')
    % Let's sample in LAB around this input color
    % Rough LAB ranges:
    % L: 0...100
    % A: -120...120
    % B: -120...120
    % We pick the gaussian scale accordingly
    delta = normrnd(10, 5, [nsamples, 1]) .* rand_sign(nsamples);  % L pertrubation
    delta(:, 2) = normrnd(20, 10, [nsamples, 1]) .* rand_sign(nsamples);  % A pertrubation
    delta(:, 3) = normrnd(20, 10, [nsamples, 1]) .* rand_sign(nsamples);  % B pertrubation
    
    lab_samples = rgb2lab(repmat(inrgb, nsamples, 1)) + delta;
    
    % Now let's convert these to RGB and only keep the valid ones
    r = lab2rgb(lab_samples);
elseif strcmp(sample_space, 'RGB')
    % Sample directly in RGB
    delta = normrnd(0.2, 0.2, [nsamples, 1]) .* rand_sign(nsamples);  % L pertrubation
    delta(:, 2) = normrnd(0.2, 0.2, [nsamples, 1]) .* rand_sign(nsamples);  % A pertrubation
    delta(:, 3) = normrnd(0.2, 0.2, [nsamples, 1]) .* rand_sign(nsamples);  % B pertrubation
    
    r = repmat(inrgb, nsamples, 1) + delta;
end

valid_rows = (r(:,1) <= 1.0 & r(:,1) >= 0.0 & r(:,2) <= 1.0 & r(:,2) >= 0.0 & r(:,3) <= 1.0 & r(:,3) >= 0.0);

rgb_samples = r(valid_rows, :);

% PLOT in RGB
plotRGB(rgb_samples);
hold on
scatter3([inrgb(1)], [inrgb(2)], [inrgb(3)], 80, inrgb, 'filled');
title('RGB');

% PLOT in LAB
lab_samples = rgb2lab(rgb_samples);
plotColors(lab_samples, rgb_samples, 40, 'L', 'A', 'B');
ax.XLim = [ 0, 100.0];
ax.YLim = [ -120, 120.0];
ax.ZLim = [ -120, 120.0];
hold on
inlab = rgb2lab(inrgb);
scatter3([inlab(1)], [inlab(2)], [inlab(3)], 80, inrgb, 'filled');
title('LAB');

% PLOT in HSV
hsv_samples = rgb2hsv(rgb_samples);
plotColors(hsv_samples, rgb_samples, 40, 'H', 'S', 'V');
%ax.XLim = [ 0, 100.0];
%ax.YLim = [ -120, 120.0];
%ax.ZLim = [ -120, 120.0];
hold on
inhsv = rgb2hsv(inrgb);
scatter3([inhsv(1)], [inhsv(2)], [inhsv(3)], 80, inrgb, 'filled');
title('HSV');

end

function res = rand_sign(nsamples)
    res = (rand(nsamples,1) > 0.5)*2 - 1
end

function plotColors(V, color, S, titleX, titleY, titleZ)
    fh = figure; 
    scatter3(V(1:end,1), V(1:end,2), V(1:end,3), S, color, 'filled'); %title('RGB')
    % set(fh, 'Position', [100, 100, 1200, 1200]);  % Uncomment for consistent scale
    ax = gca;
    ax.GridAlpha = 0.4;
    ax.GridLineStyle = ':';
    ax.XLabel.String = titleX;
    ax.YLabel.String = titleY;
    ax.ZLabel.String = titleZ;
    ax.LineWidth = 1.5;
end
