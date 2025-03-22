function h = compute3DHistogram(colors, nbins)

quant = cast(min(max(ceil(colors * nbins), 1), nbins), 'int32');
h = accumarray(quant, ones(size(quant, 1), 1), [nbins, nbins, nbins]);
h = h ./ sum(h, 'all');
end