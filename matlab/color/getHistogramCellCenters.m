function [bin_indices, bin_colors] = getHistogramCellCenters(nbins, exact_cell_centers)
[X,Y,Z] = meshgrid(1:nbins, 1:nbins, 1:nbins);
bin_indices = cat(2, X(:), Y(:), Z(:));

binsize = 1.0 / nbins;
bin_colors = (bin_indices - 0.5) * binsize;


if ~isempty(exact_cell_centers)
   red = exact_cell_centers(:, :, :, 1); 
   green = exact_cell_centers(:, :, :, 2);
   blue = exact_cell_centers(:, :, :, 3);
   idx = sub2ind(size(red), bin_indices(:, 1), bin_indices(:, 2), bin_indices(:, 3));
   bin_colors = cat(2, red(idx), green(idx), blue(idx));
end

end

