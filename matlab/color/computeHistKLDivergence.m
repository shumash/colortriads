function kl = computeHistKLDivergence(h0, h1)
%UNTITLED17 Summary of this function goes here
%   Detailed explanation goes here

min_hist_val = 1.0e-29;

p = max(h0, min_hist_val);
q = max(h1, min_hist_val);
diff = log(q) - log(p);
elems = h0 .* diff;
kl = -1.0 * sum(elems, 'all');
end

