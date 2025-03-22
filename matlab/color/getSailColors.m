function colors = getSailColors(verts, wind, nsubdiv)
inflation = wind(1)
u = wind(2)
v = wind(3)
pts = getSailControlPoints(verts, inflation, u, v);
pts = reshape(pts, [1, size(pts, 1), size(pts, 2)]);

[~, ~, bernst, ~] = bezierTriangleWeights(nsubdiv, 3);
colors = squeeze(sum(bernst .* pts, 2));
end

