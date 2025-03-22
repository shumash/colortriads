function [ img,lab_out,rgb_out, dist_out ] = sampleLabDist(dist, num)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

rgb = rand([num, 3]);
lab = rgb2lab(rgb);

% Use equal area projection to compute random vectors
theta = rand([num, 1]) * 2 * pi;
z = rand([num, 1]) * 2 - 1;
z_sq = z .* z;
vecs = sqrt(1-z_sq) .* cos(theta);
vecs(:,2) = sqrt(1-z_sq) .* sin(theta);
vecs(:,3) = z;


lab2 = lab + vecs * dist;
D = repmat(ones(num,1) * dist, 1, 3);
%D = repmat(rand([num,1 ]) * dist, 1, 3);
%lab2 = lab + vecs .* D;

rgb2 = lab2rgb(lab2);
rgb_out = zeros(1,2,3);
lab_out = zeros(1,2,3);
dist_out = zeros(1,1);
ok_count=1;
for r=1:num
   if sum(sum(rgb2(r, :)<0.0)) > 0 ||  sum(sum(rgb2(r, :)>1.0)) > 0
       rgb2(r, :) = 0.0;
       rgb(r, :) = 0.0;
   else
       rgb_out(ok_count, 1, :) = rgb(r, :);
       rgb_out(ok_count, 2, :) = rgb2(r, :);
       lab_out(ok_count,1,:) = lab(r, :);
       lab_out(ok_count,2,:) = lab2(r, :);
       dist_out(ok_count) = D(r);
       ok_count = ok_count + 1;
   end
end

img = zeros(num, 2, 3);
img(:,1,:) = rgb;
img(:,2,:) = rgb2;

% lab_out = zeros(num, 2, 3);
% lab_out(:,1,:) = lab;
% lab_out(:,2,:) = lab2;

end

