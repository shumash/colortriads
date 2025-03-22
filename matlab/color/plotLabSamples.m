function [ ] = plotLabSamples(random_lab_rgb, file_lab)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

M=importdata(file_lab);
sz = size(M);
LM=zeros([sz(1), 6]);
LM(:,4:6) = M(:,1:3);
LM(:,1:3) = lab2rgb(M(:,1:3));
r = random_lab_rgb;
figure;
if ~isempty(random_lab_rgb)
     scatter3(r(:,4), r(:,5), r(:,6), 40, r(:,1:3), 'filled')
    hold on;
end
scatter3(LM(:,4), LM(:,5), LM(:,6), 40, 'black');

end

