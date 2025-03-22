function h = computeSoft3DHistogram(colors, nbins)

sigma_sq = (0.5 / nbins) ^ 2;
[bin_indices, bin_colors] = getHistogramCellCenters(nbins, []);

% Next... sum up all contributions for a cell
D = ipdm(bin_colors, colors);
D = exp( (D .^ 2) ./ (-2 * sigma_sq) );
D = sum(D, 2);
D = D ./ sum(D, 'all');

h = accumarray(bin_indices, D, [nbins, nbins, nbins]);
end

