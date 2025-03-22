function [ coords ] = plotGradient(filename, sampleInterval, asLine, inLab)
% Plots a gradient stored in a horizontal image (color changing
% horizontally); sampleInterval is how far apart in the horizontal
% direction to sample; if asLine plots as a smooth line; otherwise as dots.

I = imread(filename);
sm = sampleHorizontalGradient(I, sampleInterval);
coords = sm;
size(sm)
if inLab
    coords = rgb2lab(sm);
end
if asLine
    r = sm(:, 1);
    g = sm(:, 2);
    b = sm(:, 3);
    x = coords(:, 1);
    y = coords(:, 2);
    z = coords(:, 3);
    C(:, :, 1) = [r(:), r(:)];
    C(:, :, 2) = [g(:), g(:)];
    C(:, :, 3) = [b(:), b(:)];
    surf([x(:), x(:)], [y(:), y(:)], [z(:), z(:)], C,...
        'facecol','no', 'EdgeColor','interp', 'linew', 5);
else
    scatter3(coords(:, 1), coords(:, 2), coords(:, 3), 40, sm, 'filled');
end

end

