function plot3DHistogram(h, exact_cell_centers)
% h: nxnxn normalized counts

[bin_indices, bin_colors] = getHistogramCellCenters(size(h, 1), exact_cell_centers);

max_size = 420;

idx = sub2ind(size(h), bin_indices(:, 1), bin_indices(:, 2), bin_indices(:, 3));
sz_prev = h(:) .* (max_size / max(h(:))) + 0.01;
sz = h(idx) .* (max_size / max(h(:))) + 0.01;

scatter3(bin_colors(:, 1), bin_colors(:, 2), bin_colors(:, 3), sz, bin_colors, 'filled');

end

