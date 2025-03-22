function [pts, pts_no_wind] = getSailControlPoints(colors, inflation, u, v)
w = 1 - u - v;

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

p300 = u300 * colors;
p003 = u003 * colors;
p030 = u030 * colors;
p012 = u012 * colors;
p021 = u021 * colors;
p102 = u102 * colors;
p201 = u201 * colors;
p120 = u120 * colors;
p210 = u210 * colors;
p111 = u111 * colors;

dist_sq = [ 1000,... % 300
            1000,... % 030
            1000,... % 003
            2.0 * u * u,... % 012
            2.0 * u * u,... % 021
            2.0 * v * v,... % 102
            2.0 * v * v,... % 201
            2.0 * ((1.0 - u - v)^2),... % 120
            2.0 * ((1 - u - v)^2),... % 210
            0]; % 111
normal = cross(p300 - p030, p300 - p003);
normal = normal * inflation;
normal = reshape(normal, [3,1]);
pressure = exp(dist_sq * (-1.0 / 0.8));

influence = (normal * pressure)';

pts = [p300; p030; p003;
       p012; p021; p102;
       p201; p120; p210; p111];
pts_no_wind = pts;
pts = pts_no_wind + influence;
    
end

