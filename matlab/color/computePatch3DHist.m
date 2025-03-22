function [h, exact_cell_centers] = computePatch3DHist(in_img, img_width, patch_width, nbins, normalize)

%img_width = 512;
%patch_width = 32;

img = im2double(imresize(in_img, [img_width img_width]));
%colors = reshape(img, [img_width * img_width, 3]);
quant = cast(min(max(ceil(img * nbins), 1), nbins), 'int32');
one_counts = ones(patch_width * patch_width, 1);

npatches = floor(img_width / patch_width);
h = zeros(nbins, nbins, nbins);
h_counts = 0;
r_sum = 0;
g_sum = 0;
b_sum = 0;
for x=1:npatches-1
    for y=1:npatches-1
        pstartx = x * patch_width;
        pstarty = y * patch_width;
        q = quant(pstartx:(pstartx+patch_width-1),pstarty:(pstarty+patch_width-1), :);
        q = reshape(q, [patch_width * patch_width, 3]);  % why needed?
        new_h = accumarray(q, one_counts, [nbins, nbins, nbins]);
        h = max(h, new_h);
        h_counts = h_counts + new_h;
        
        patch = img(pstartx:(pstartx+patch_width-1),pstarty:(pstarty+patch_width-1), :);
        r_sum = r_sum + accumarray(q, reshape(patch(:,:,1), [patch_width * patch_width, 1]), [nbins, nbins, nbins]);
        g_sum = g_sum + accumarray(q, reshape(patch(:,:,2), [patch_width * patch_width, 1]), [nbins, nbins, nbins]);
        b_sum = b_sum + accumarray(q, reshape(patch(:,:,3), [patch_width * patch_width, 1]), [nbins, nbins, nbins]);
    end
end


if normalize
    h = h ./ sum(h, 'all');
end
h_counts(h_counts < 0.001) = 1;
exact_cell_centers = cat(4, reshape(r_sum ./ h_counts, [nbins, nbins, nbins, 1]), ...
    reshape(g_sum ./ h_counts, [nbins, nbins, nbins, 1]),...
    reshape(b_sum ./ h_counts, [nbins, nbins, nbins, 1]));

end

