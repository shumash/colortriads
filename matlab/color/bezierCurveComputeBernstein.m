function [res] = bezierCurveComputeBernstein(order)
%UNTITLED4 returns matrix of bernstein coefficients B, such that
%
% [(1-t0)^n (1-t0)^(n-1)...] * B * [R0 G0 B0]                          
% [(1-t1)^n (1-t1)^(n-1)...]       [R1 G1 B1] = interpolated pts at t0, t1
% [   ...                  ]       [  ...   ]
% 
% Order is the number of control pts - 1.
%

res = eye(order+1);
for i=1:order+1
    res(i, i) = nchoosek(order, i-1);
end

end

