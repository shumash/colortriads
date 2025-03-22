function [cpts] = bezierCurveOptimize(order, firstPt, lastPt, targetPts)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

% Optimizing for the following parameters (N==number of target Pts):
% N time values
% order -1 control points

Ntargets = size(targetPts, 1);
Nvars = Ntargets + 3 * (order - 1);
tstart = linspace(1.0 / Ntargets, 1.0 - 1.0 / Ntargets, Ntargets);
bernPts = bezierCurveComputeBernstein(order);

alpha = linspace(1.0 / order, 1.0 - 1.0 / order, order - 1);
cstart_colors = alpha' * firstPt + (1 - alpha') * lastPt;
cstart = reshape(cstart_colors, [1, 3 * (order - 1)]);

figure;
scatter3(targetPts(:, 1), targetPts(:, 2), targetPts(:, 3), 40, targetPts, 'filled');

x0 = cat(2, tstart, cstart);
lb = zeros(1, Nvars);
ub = ones(1, Nvars);

[loss, guessPts] = lossFunctionUtil(order, firstPt, lastPt, targetPts, bernPts, x0);
hold on;
scatter3(guessPts(:, 1), guessPts(:, 2), guessPts(:, 3), 20, [0.8, 0.8, 0.8], 'filled');

x = fmincon(@(x)lossFunction(order, firstPt, lastPt, targetPts, bernPts, x), ...
    x0, [], [], [], [], lb, ub)

[loss, approxPts, cpts, t] = lossFunctionUtil(order, firstPt, lastPt, targetPts, bernPts, x);
hold on;
scatter3(approxPts(:, 1), approxPts(:, 2), approxPts(:, 3), 20, approxPts, 'filled');

end


function loss = lossFunction(order, firstPt, lastPt, targetPts, bernPts, x)

[loss, ~, ~, ~] = lossFunctionUtil(order, firstPt, lastPt, targetPts, bernPts, x);
loss

end

function [loss, approxPts, cpts, t] = lossFunctionUtil(order, firstPt, lastPt, targetPts, bernPts, x)

Ntargets = size(targetPts, 1);

t = x(1, 1:Ntargets);
cpts = cat(1, firstPt, reshape(x(Ntargets+1:end), [(order - 1), 3]), lastPt);

approxPts = bezierCurveComputeTime(order, t) * bernPts * cpts;

loss = sum(sum((approxPts - targetPts).^2));

end
