function [ n, pts_out, tri_pts_out ] = bezierTriangle(v0, v1, v2, inflation, u, v, nsubdiv) %p300, p030, p003, inflation)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

%v = 0.2; %1/3.0;
w = 1 - u - v;
in = [v0; v1; v2];

u300 = [1, 0, 0];
u003 = [0, 0, 1];
u030 = [0, 1, 0];
u012 = [0, v, 1-v]; 
u021 = [0, u+v, 1-u-v];
u102 = [u, 0, 1-u];
u201 = [u+v, 0, 1-u-v];
u120 = [u, 1-u, 0]; 
u210 = [1-v, v, 0];
u111 = [u, v, w];

canonical_pos = [ -0.5, 0.0, 0.0; 0.5, 0.0, 0.0; 0.0, sqrt(3.0) * 0.5, 0 ];
canonical_pos(:, 1) = canonical_pos(:, 1) + 0.5;
n = 3;
normal = normalizerow(cross(v0 - v1, v0 - v2));
ave_side = (norm(v0 - v1) + norm(v0 - v2) + norm(v1 - v2)) / 3.0;
sigma = ave_side;

% p300 = v0;
% p030 = v1;
% p003 = v2;
% p012 = p003 + (p030 - p003) / 3.0;
% p021 = p003 + (p030 - p003) / 3.0 * 2.0;
% p102 = p003 + (p300 - p003) / 3.0;
% p201 = p003 + (p300 - p003) / 3.0 * 2.0;
% p120 = p030 + (p300 - p030) / 3.0;
% p210 = p030 + (p300 - p030) / 3.0 * 2.0;
% p111 = (p300 + p003 + p030) / 3.0; % + inflation * normal;
% ppps_prev = [p300; p003; p030; p012; p021; p102; p201; p120; p210; p111];

p300 = u300 * in;
p003 = u003 * in;
p030 = u030 * in;
p012 = u012 * in;
p021 = u021 * in;
p102 = u102 * in;
p201 = u201 * in;
p120 = u120 * in;
p210 = u210 * in;
p111 = u111 * in;


ppps = [p300; p003; p030; p012; p021; p102; p201; p120; p210; p111];
ppps_can = [ ...
u300 * canonical_pos; ...
u003 * canonical_pos; ...
u030 * canonical_pos; ...
u012 * canonical_pos; ...
u021 * canonical_pos; ...
u102 * canonical_pos; ...
u201 * canonical_pos; ...
u120 * canonical_pos; ...
u210 * canonical_pos; ...
u111 * canonical_pos ];

% add slight inflation to the other points
p012 = p012 + exp(-0.5 * power(norm(p012 - p111)/sigma, 2)) * inflation * normal;
p021 = p021 + exp(-0.5 * power(norm(p021 - p111)/sigma, 2)) * inflation * normal;
p102 = p102 + exp(-0.5 * power(norm(p102 - p111)/sigma, 2)) * inflation * normal;
p201 = p201 + exp(-0.5 * power(norm(p201 - p111)/sigma, 2)) * inflation * normal;
p120 = p120 + exp(-0.5 * power(norm(p120 - p111)/sigma, 2)) * inflation * normal;
p210 = p210 + exp(-0.5 * power(norm(p210 - p111)/sigma, 2)) * inflation * normal;
p111 = p111 + inflation * normal;

pts = [p300; p030; p003;
       p012; p021; p102;
       p201; p120; p210; p111];
pts = reshape(pts, [1, size(pts, 1), size(pts, 2)]);

naive_pos = [[ 0.        ,  0.        ],
       [ 0.25      ,  0.        ],
       [ 0.125     ,  0.21650635],
       [ 0.25      ,  0.        ],
       [ 0.5       ,  0.        ],
       [ 0.375     ,  0.21650635],
       [ 0.125     ,  0.21650635],
       [ 0.375     ,  0.21650635],
       [ 0.25      ,  0.43301269],
       [ 0.25      ,  0.        ],
       [ 0.375     ,  0.21650635],
       [ 0.125     ,  0.21650635],
       [ 0.        ,  0.        ],
       [ 0.125     ,  0.        ],
       [ 0.0625    ,  0.10825317],
       [ 0.125     ,  0.        ],
       [ 0.25      ,  0.        ],
       [ 0.1875    ,  0.10825317],
       [ 0.0625    ,  0.10825317],
       [ 0.1875    ,  0.10825317],
       [ 0.125     ,  0.21650635],
       [ 0.25      ,  0.        ],
       [ 0.375     ,  0.        ],
       [ 0.3125    ,  0.10825317],
       [ 0.375     ,  0.        ],
       [ 0.5       ,  0.        ],
       [ 0.4375    ,  0.10825317],
       [ 0.3125    ,  0.10825317],
       [ 0.4375    ,  0.10825317],
       [ 0.375     ,  0.21650635],
       [ 0.125     ,  0.21650635],
       [ 0.25      ,  0.21650635],
       [ 0.1875    ,  0.32475951],
       [ 0.25      ,  0.21650635],
       [ 0.375     ,  0.21650635],
       [ 0.3125    ,  0.32475951],
       [ 0.1875    ,  0.32475951],
       [ 0.3125    ,  0.32475951],
       [ 0.25      ,  0.43301269],
       [ 0.25      ,  0.        ],
       [ 0.3125    ,  0.10825317],
       [ 0.1875    ,  0.10825317],
       [ 0.3125    ,  0.10825317],
       [ 0.375     ,  0.21650635],
       [ 0.25      ,  0.21650635],
       [ 0.1875    ,  0.10825317],
       [ 0.25      ,  0.21650635],
       [ 0.125     ,  0.21650635]];
   
 bary_naive = [[ 0.66666669,  0.16666667,  0.16666667],
       [ 0.16666667,  0.66666669,  0.16666667],
       [ 0.16666667,  0.16666667,  0.66666669],
       [ 0.33333334,  0.33333334,  0.33333334],
       [ 0.83333331,  0.08333334,  0.08333334],
       [ 0.58333331,  0.33333334,  0.08333334],
       [ 0.58333331,  0.08333334,  0.33333334],
       [ 0.33333334,  0.58333331,  0.08333334],
       [ 0.08333334,  0.83333331,  0.08333334],
       [ 0.08333334,  0.58333331,  0.33333334],
       [ 0.33333334,  0.08333334,  0.58333337],
       [ 0.08333334,  0.33333334,  0.58333331],
       [ 0.08333334,  0.08333334,  0.83333331],
       [ 0.41666669,  0.41666669,  0.16666667],
       [ 0.16666667,  0.41666666,  0.41666669],
       [ 0.41666666,  0.16666667,  0.41666666]];
   
%cw = cw5; %10; %cw5; %cw5; %10; %cw4;
%bary = bary5; %10; %bary5; %bary5; %10; %bary4; %bary_naive;

[bary, cw, bernst] = bezierTriangleWeights(nsubdiv);
%[ p300; p030; p003 ]
% input = [ p300; p030; p003 ];
%pts_out = bary * in;
tri_pts_out = cw * in;
canonical_tri_pts_out = cw * canonical_pos;



[pts, ~] = getSailControlPoints(in, inflation, u, v);
pts = reshape(pts, [1, size(pts, 1), size(pts, 2)]);
pts_out = squeeze(sum(bernst .* pts, 2));
size(pts_out)


p300 = pts(1, 1, :);
p030 = pts(1, 2, :);
p003 = pts(1, 3, :);
p012 = pts(1, 4, :);
p021 = pts(1, 5, :);
p102 = pts(1, 6, :);
p201 = pts(1, 7, :);
p120 = pts(1, 8, :);
p210 = pts(1, 9, :);
p111 = pts(1, 10, :);
for p=1:size(tri_pts_out, 1)
    tv = cw(p, 1);
    tu = cw(p, 2);
    tw = 1 - tv - tu;
    v = tu;
    u = tv;
    % Needed, because these are for different u, v's than original sail
    pos = computeBernsteinPos(u, v, n, p300, p030, p003, ...
        p012, p021, p102, p201, p120, p210, p111);
    tri_pts_out(p, :) = pos; 
end


[~, pts_no_wind] = getSailControlPoints(canonical_pos, 0.0, 1/3.0, 1/3.0);
p300 = pts_no_wind(1, :);
p030 = pts_no_wind(2, :);
p003 = pts_no_wind(3, :);
p012 = pts_no_wind(4, :);
p021 = pts_no_wind(5, :);
p102 = pts_no_wind(6, :);
p201 = pts_no_wind(7, :);
p120 = pts_no_wind(8, :);
p210 = pts_no_wind(9, :);
p111 = pts_no_wind(10, :);


for p=1:size(canonical_tri_pts_out, 1)
    tv = cw(p, 1);
    tu = cw(p, 2);
    tw = 1 - tv - tu;
    v = tu;
    u = tv;
    % Needed, because these are for different u, v's than original sail
    pos = computeBernsteinPos(u, v, n, p300, p030, p003, ...
        p012, p021, p102, p201, p120, p210, p111);
    canonical_tri_pts_out(p, :) = pos; 
end
disp('Can tri pts')
size(canonical_tri_pts_out)
figure;
plotTris(canonical_tri_pts_out, pts_out);
title('Flat')
pbaspect([1,1,1])
view(0,90)
grid off
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])
set(gca,'visible','off')

%figure; 
%plotTris(tri_pts_out, pts_out);
% plotTris(cw * canonical_pos, pts_out);
% hold on
% tmp = ppps_can + [0.0, 0.0, 0.1];
% scatter3(tmp(:,1), tmp(:,2), tmp(:,3), 70, 'black', 'filled');
% title('With Wind')

figure; 
plotTris(tri_pts_out, pts_out);
title('3D')
%hold on
%tmp = ppps;
%scatter3(tmp(:,1), tmp(:,2), tmp(:,3), 70, 'red', 'filled');
%figure;
%scatter3(pts_out(:,1), pts_out(:,2), pts_out(:,3), 70, pts_out, 'filled'); title('interp')

tri_pts_out = bary * canonical_pos;

end

function plotTris(v, colors)
% v: (ntri * 3)  x 3
% colors: (ntri) x 3

R = v(:, 1);
G = v(:, 2);
B = v(:, 3);
X = max(0.0, min(1.0, reshape(R', [3, size(R,1)/3])));
Y = max(0.0, min(1.0, reshape(G', [3, size(G,1)/3])));
Z = max(0.0, min(1.0, reshape(B', [3, size(B,1)/3])));
colors = max(0.0, min(1.0, colors));

for i = 1:size(colors,1)
   fill3(X(:, i), Y(:, i), Z(:, i), colors(i, :)); %, 'LineStyle','none');
   hold on;
end
xlim([0, 1.0]);
ylim([0, 1.0]);
zlim([0, 1.0]);
grid on
end

function pos = computeBernsteinPos(u, v, n, p300, p030, p003, ...
       p012, p021, p102, p201, p120, p210, p111)
   pos = computeBernstein(n, 3, 0, 0, u, v) * p300 + ...
       computeBernstein(n, 0, 3, 0, u, v) * p030 + ...
       computeBernstein(n, 0, 0, 3, u, v) * p003 + ...
       computeBernstein(n, 0, 1, 2, u, v) * p012 + ...
       computeBernstein(n, 0, 2, 1, u, v) * p021 + ...
       computeBernstein(n, 1, 0, 2, u, v) * p102 + ...
       computeBernstein(n, 2, 0, 1, u, v) * p201 + ...
       computeBernstein(n, 1, 2, 0, u, v) * p120 + ...
       computeBernstein(n, 2, 1, 0, u, v) * p210 + ...
       computeBernstein(n, 1, 1, 1, u, v) * p111;
end

function res = computeBernstein(n, i, j, k, u, v)
    res = factorial(n) / (factorial(i) * factorial(j) * factorial(k)) * power(u, i) * power(v, j) * power(1.0 - u - v, k);
end


% c0 =
% 
%     0.3608    0.9529    0.9647
% 
% c1
% 
% c1 =
% 
%     0.1098    0.6235    0.9882
% 
% c2
% 
% c2 =
% 
%     0.9961    0.7765    0.1804

% Purple blue yellow
% cc0 =
% 
%     0.4118    0.0706    0.5725
% 
% cc1
% 
% cc1 =
% 
%     0.9647    0.8863    0.1961
% 
% cc2
% 
% cc2 =
% 
%     0.0549    0.2863    0.8667

