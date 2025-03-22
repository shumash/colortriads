function [ res ] = sampleRGBToLab(n_samples)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

res = double(zeros(n_samples, 6));
for i=1:n_samples
    rgb = rand([1, 3]);
    res(i, 1:3) = rgb;
    lab = rgb2lab(rgb, 'WhitePoint','d50');
    res(i, 4:6) = lab;
end

end

% TO PLOT: scatter3(r(:,4), r(:,5), r(:,6), 40, r(:,1:3), 'filled')