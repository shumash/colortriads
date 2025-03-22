function corner_colors = colorSailGuessCornerColors(target_colors)
mu = mean(target_colors, 1);
d = ipdm(target_colors, mu);

% First point is farthest point from the centroid
[~, midx] = max(d);
p0 = target_colors(midx, :);

% Then, we find a point furthest from p0
dp0 = ipdm(target_colors, p0);
[~, midx] = max(dp0);
p1 = target_colors(midx, :);

% Then, we find distance of all points to this p1 as well
dp1 = ipdm(target_colors, p1);

% We want a point far from both p0 and p1, so let's multiply distances
d = dp0 .* dp1;
[~, midx] = max(d);
p2 = target_colors(midx, :);

corner_colors = [p0; p1; p2];
end

