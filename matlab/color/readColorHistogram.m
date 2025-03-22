function [ H ] = readColorHistogram(filename, graph)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
D = dlmread(filename);
H = zeros(D(1,1), D(1,2), D(1,3));

idx = sub2ind(size(H), uint32(D(2:end,1) + 1), uint32(D(2:end,2) + 1), uint32(D(2:end,3) + 1));
H(idx) = D(2:end, 4);

RGB = double(D(2:end, 1)) / D(1,1);
RGB(:,2) = double(D(2:end, 2)) / D(1,2);
RGB(:,3) = double(D(2:end, 3)) / D(1,3);

if graph
   s = sum(D(2:end, 4));
   fracs = D(2:end, 4) / s;
   mi = min(fracs);
   ma = max(fracs);
   min_size = 10;
   max_size = 200;
   sz = (fracs - mi) / (ma - mi);  % 0..1
   sz = min_size + sz * (max_size - min_size);
   
%    figure;
%    scatter3(D(2:end, 1), D(2:end, 2), D(2:end, 3), sz, D(2:end, 11:13), 'filled');
%    title('Orig hist');
   
   figure;
   scatter3(D(2:end, 1), D(2:end, 2), D(2:end, 3), sz, RGB, 'filled');
   title('Computed color hist');
end
end

