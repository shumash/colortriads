function plotRGB(V)
%PLOTRGB Plots a matrix of colors as a scatter plot
%   V has one color per row, with r, g, b values (0-1) in cols 1-3
    if size(V, 2) == 4
        S = 40 + 400 * double(V(:, 4) - min(V(:, 4))) / double(max(1, max(V(:, 4)) - min(V(:, 4))));
    else
        S = 40;
    end
    % fh = figure; 
    scatter3(V(1:end,1), V(1:end,2), V(1:end,3), S, V(1:end,1:3), 'filled'); %title('RGB')
    % set(fh, 'Position', [100, 100, 1200, 1200]);  % Uncomment for consistent scale
    ax = gca;
    ax.GridAlpha = 0.4;
    ax.GridLineStyle = ':';
    ax.XLabel.String = 'R';
    ax.YLabel.String = 'G';
    ax.ZLabel.String = 'B';
    ax.XLim = [ 0, 1.0]
    ax.YLim = [ 0, 1.0]
    ax.ZLim = [ 0, 1.0]
    ax.LineWidth = 1.5;


end
