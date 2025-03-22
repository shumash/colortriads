function [ img ] = vizLabDelta(nsamples, delta, width)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

rgblab = sampleRGBToLab(nsamples);

pair_rgblab = [];
% For each lab sample, pick a random vector, 
for i = 1:size(rgblab,1)
    theta = rand() * 2 * pi;
    z = rand() * 2 - 1;
    vec = [sqrt(1 - z*z) * cos(theta), sqrt(1 - z*z) * sin(theta), z];
    newlab = rgblab(i, 4:6) + vec * delta;
    newrgb = lab2rgb(newlab, 'WhitePoint','d50');
    if max(newrgb(:)) <= 1.0 && min(newrgb(:)) >= 0.0
        if size(pair_rgblab, 1) == 0
            pair_rgblab = [rgblab(i,:), newrgb, newlab];
        else
            pair_rgblab = [pair_rgblab; rgblab(i,:), newrgb, newlab];
        end
    end
end

orig = rgb2image(pair_rgblab(:,1:3), width);
moved = rgb2image(pair_rgblab(:,7:9), width);
img = cat(2, orig, moved);

end

function img = rgb2image(values, w)

R = values(:, 1);
G = values(:, 2);
B = values(:, 3);

Rs = reshape(repmat(R', w*w, 1), [w,w*numel(R)])';
Gs = reshape(repmat(G', w*w, 1), [w,w*numel(G)])';
Bs = reshape(repmat(B', w*w, 1), [w,w*numel(B)])';

img(:,:,1) = Rs;
img(:,:,2) = Gs;
img(:,:,3) = Bs;

end