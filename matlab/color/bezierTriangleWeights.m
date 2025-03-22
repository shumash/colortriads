function [weights, cweights, bernstein, cbernstein] = bezierTriangleWeights(n_tri_subdivs, n)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

if n ~=2 && n ~=3
    error('Only 2 and 3 order bezier triangles supported')
end

n_tri_colors = n_tri_subdivs * n_tri_subdivs;
n_tri_rows = 2 * n_tri_subdivs - 1;

weights = zeros(3, n_tri_colors);
cweights = zeros(3, n_tri_colors * 3);

color_count = 0;

for r = 0:(n_tri_rows-1)
    alpha = r / (n_tri_rows - 1.0);
    n_cols = n_tri_rows - (r / 2) * 2;
    start_col = mod(r, 2);
    for c = start_col:2:(n_cols - start_col)
        if mod(r, 2) == 0
            if n_cols == 1
                beta = 0.5;
            else
                beta = c / (n_cols - 1.0);
            end
            weights(3, color_count + 1) = alpha;
            weights(2, color_count + 1) = (1 - alpha) * beta;
            weights(1, color_count + 1) = (1 - alpha) * (1 - beta);
            %right_side_up_indicator[color_count] = True
        else
            % Just average neighboring weights
            % bottom left neighbor
            w0 = compute_weights(r - 1, c - 1, n_tri_rows);
            w1 = compute_weights(r - 1, c + 1, n_tri_rows);
            w2 = compute_weights(r + 1, c - 1, n_tri_rows);
            weights(3, color_count + 1) = (w0(1) + w1(1) + w2(1)) / 3.0;
            weights(2, color_count + 1) = (w0(2) + w1(2) + w2(2)) / 3.0;
            weights(1, color_count + 1) = (w0(3) + w1(3) + w2(3)) / 3.0;
        end
            

        % coordinates for vizualization------------
        viz_row = floor(r / 2) + 0.0;
        if mod(r, 2) == 0  % right side up
            % Lower left
            calpha = (viz_row + 0.0) / n_tri_subdivs;
            cbeta = floor(c/2) / max(1.0, n_tri_subdivs - viz_row);
            cweights(3, color_count * 3 + 1) = calpha;
            cweights(2, color_count * 3 + 1) = (1 - calpha) * cbeta;
            cweights(1, color_count * 3 + 1) = (1 - calpha) * (1 - cbeta);
            % Lower right
            calpha = (viz_row + 0.0) / n_tri_subdivs;
            cbeta = (floor(c/2) + 1.0) / max(1.0, n_tri_subdivs - viz_row);
            cweights(3, color_count * 3 + 1 + 1) = calpha;
            cweights(2, color_count * 3 + 1 + 1) = (1 - calpha) * cbeta;
            cweights(1, color_count * 3 + 1 + 1) = (1 - calpha) * (1 - cbeta);
            % Top
            calpha = (viz_row + 1.0) / n_tri_subdivs;
            cbeta = (floor(c/2) + 0.0) / max(1.0, n_tri_subdivs - viz_row - 1);
            cweights(3, color_count * 3 + 2 + 1) = calpha;
            cweights(2, color_count * 3 + 2 + 1) = (1 - calpha) * cbeta;
            cweights(1, color_count * 3 + 2 + 1) = (1 - calpha) * (1 - cbeta);
        else
            % Upper left
            calpha = (viz_row + 1.0) / n_tri_subdivs;
            cbeta = floor(c/2) / max(1.0, n_tri_subdivs - viz_row - 1);
            cweights(3, color_count * 3 + 1) = calpha;
            cweights(2, color_count * 3 + 1) = (1 - calpha) * cbeta;
            cweights(1, color_count * 3 + 1) = (1 - calpha) * (1 - cbeta);
            % Upper right
            calpha = (viz_row + 1.0) / n_tri_subdivs;
            cbeta = (floor(c/2) + 1.0) / max(1.0, n_tri_subdivs - viz_row - 1);
            cweights(3, color_count * 3 + 1 + 1) = calpha;
            cweights(2, color_count * 3 + 1 + 1) = (1 - calpha) * cbeta;
            cweights(1, color_count * 3 + 1 + 1) = (1 - calpha) * (1 - cbeta);
            % Bottom
            calpha = (viz_row + 0.0) / n_tri_subdivs;
            cbeta = (floor(c/2) + 1.0) / max(1.0, n_tri_subdivs - viz_row);
            cweights(3, color_count * 3 + 2 + 1) = calpha;
            cweights(2, color_count * 3 + 2 + 1) = (1 - calpha) * cbeta;
            cweights(1, color_count * 3 + 2 + 1) = (1 - calpha) * (1 - cbeta);
        end
        color_count = color_count + 1;
    end
end

weights = weights';
cweights = cweights';

if n == 3
    bernstein = computeBernsteinMat(weights(:,1), weights(:, 2));
    cbernstein = computeBernsteinMat(cweights(:,1), cweights(:, 2));
elseif n == 2
    bernstein = computeBernsteinMatQuad(weights(:,1), weights(:, 2));
    cbernstein = computeBernsteinMatQuad(cweights(:,1), cweights(:, 2));
end
    
bernstein = reshape(bernstein, [size(bernstein, 1), size(bernstein, 2), 1]);
cbernstein = reshape(cbernstein, [size(cbernstein, 1), size(cbernstein, 2), 1]);
end


function [res] = compute_weights(r, c, n_tri_rows)
alpha = r / (n_tri_rows - 1.0);
n_cols = n_tri_rows - (r / 2) * 2;
if n_cols == 1
    beta = 0.5;
else
    beta = c / (n_cols - 1.0);
end
res = [alpha, (1 - alpha) * beta, (1 - alpha) * (1 - beta)];
end
    
function res = computeBernsteinMat(u, v)
n = 3;
res = [computeBernstein(n, 3, 0, 0, u, v), ...
    computeBernstein(n, 0, 3, 0, u, v), ...
    computeBernstein(n, 0, 0, 3, u, v), ...
    computeBernstein(n, 0, 1, 2, u, v), ...
    computeBernstein(n, 0, 2, 1, u, v), ...
    computeBernstein(n, 1, 0, 2, u, v), ...
    computeBernstein(n, 2, 0, 1, u, v), ...
    computeBernstein(n, 1, 2, 0, u, v), ...
    computeBernstein(n, 2, 1, 0, u, v), ...
    computeBernstein(n, 1, 1, 1, u, v)];
end

function res = computeBernsteinMatQuad(u, v)
n = 2;
% p200; p020; p002; p011; p101; p110
res = [computeBernstein(n, 2, 0, 0, u, v), ...
    computeBernstein(n, 0, 2, 0, u, v), ...
    computeBernstein(n, 0, 0, 2, u, v), ...
    computeBernstein(n, 0, 1, 1, u, v), ...
    computeBernstein(n, 1, 0, 1, u, v), ...
    computeBernstein(n, 1, 1, 0, u, v)];
end
