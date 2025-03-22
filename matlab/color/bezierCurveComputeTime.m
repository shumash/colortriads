function [res] = bezierCurveComputeTime(order, time_values)
%UNTITLED5 Returns an M(n-tvalues) x N(order+1) matrix 
% such as:
% [(1-t0)^n (1-t0)^(n-1)...]                       
% [(1-t1)^n (1-t1)^(n-1)...] 
% [       ...              ]

tvec = reshape(time_values, [numel(time_values), 1]);
tvec_oneminus = 1 - tvec;
res = ones(numel(time_values), order + 1);
for i=2:order+1
    res(:, i) = tvec .* res(:, i-1);
end

res2 = ones(numel(time_values), order + 1);
for i=order:-1:1
   res2(:, i) = tvec_oneminus .* res2(:, i+1);
end
res = res.* res2;

end

