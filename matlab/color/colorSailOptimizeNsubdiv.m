function [colors, wind, nsubdiv, meta] = colorSailOptimizeNsubdiv(input_colors, input_wind, target_colors, opts, max_nsubdiv)
% Find smallest nsubdiv s.t. loss is less than max_degradation bigger than
% starter loss.

max_degradation = 0.0025; %0.025;

losses = -1.0 * ones(max_nsubdiv, 1);


[colors, wind, ~, submeta] = colorSailOptimize(input_colors, input_wind, target_colors, opts, max_nsubdiv);
meta(max_nsubdiv) = submeta(1);
sails(max_nsubdiv).colors = colors;
sails(max_nsubdiv).wind = wind;

best_loss = submeta(1).losses(end);
max_loss = best_loss + max_degradation;
disp(sprintf('Max loss: %0.4f', max_loss));
losses(max_nsubdiv) = best_loss;

% top: always acceptable
% bottom: always not acceptable
top = max_nsubdiv;
bottom = 1;
while top > bottom + 1
    step = floor((top - bottom) / 2);
    n = bottom + step;
    
    [colors, wind, ~, submeta] = colorSailOptimize(input_colors, input_wind, target_colors, opts, n);
    meta(n) = submeta(1);
    sails(n).colors = colors;
    sails(n).wind = wind;
    losses(n) = submeta(1).losses(end);
    
    disp(sprintf('(bottom = %0.1f, top = %0.1f, n = %0.1f (loss %0.4f vs. %0.4f max)', bottom, top, n, losses(n), max_loss));
    
    if losses(n) > max_loss
        bottom = n;
    else
        top = n;
    end 
end

nsubdiv = top;
colors = sails(nsubdiv).colors;
wind = sails(nsubdiv).wind;
losses

end

% NAIVE
% for n=2:max_nsubdiv
%     [colors, wind, ~, submeta] = colorSailOptimize(input_colors, wind, target_colors, opts, n)
%     meta(n) = submeta(1);
%     sails(n).colors = colors;
%     sails(n).wind = wind;
%     losses(n) = submeta(1).losses(end);
%     losses
% end
% 
% nsubdiv = 2;
