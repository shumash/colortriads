function res = drawPlaneNVP(n, v, p)

[X,Y] = meshgrid(-1:0.1:1);

res = zeros(3,3,3);
for i = 1:max(size(X))
    for j = 1:max(size(Y))
       res(i,j,:) = (p' + X(i,j) * v(:,1) + Y(i,j) * v(:, 2))';
    end
end
end

%surf(res(:,:,1), res(:,:,2), res(:, :,3), res, 'EdgeColor', 'none')
%shading interp
%