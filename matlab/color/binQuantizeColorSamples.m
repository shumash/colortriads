function out_colors = binQuantizeColorSamples(colors, numbins, precise, subsets)
% N x 3 RGB colors (float 0..1)
% numbins -- how many quant bins to use (e.g. 25)

% With permuttion
%p = randperm(size(colors, 1));
%A = cast(colors(p, :) * numbins, 'uint8');
%orig_colors = colors(p, :);

A = cast(colors * numbins, 'uint8');
orig_colors = colors;

out_colors = getUnique(orig_colors, A, numbins, precise);
if subsets > 1
    subsetsize = floor(size(A, 1) / subsets);
    for s=0:(subsets - 1)
        A_sub = A((s * subsetsize + 1):(s * subsetsize + subsetsize), :);
        colors_sub = orig_colors((s * subsetsize + 1):(s * subsetsize + subsetsize), :);
        out_colors = [out_colors; getUnique(colors_sub, A_sub, numbins, precise)];
    end
end
end

function out_colors = getUnique(orig_colors, A, numbins, precise)
[u, ia, ic] = unique(A, 'rows');
    out_colors = cast(u, 'double') / numbins;

if precise
    for i = 1:numel(ia)
       out_colors(i, :) = mean(orig_colors(ic == i, :), 1); 
    end
end

end


