function [res] = sampleHorizontalGradient(img, interval)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

res = im2double(squeeze(img(size(img,1)/2, 1:interval:end, :)));

end

