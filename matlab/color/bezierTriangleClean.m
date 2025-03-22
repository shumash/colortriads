function [ colors_out, control_pts ] = bezierTriangleClean(in_pts, wind, nsubdiv) %p300, p030, p003, inflation)

% Flat 2D positions
canonical_pos = [ -0.5, 0.0, 0.0; 0.5, 0.0, 0.0; 0.0, sqrt(3.0) * 0.5, 0 ];
canonical_pos(:, 1) = canonical_pos(:, 1) + 0.5;

if size(in_pts, 1) == 6
    %                 p200
    %
    %
    %        p110                p101
    %
    %
    %  p020           p011              p002
    disp('Rendering quadratic bezier triangle')
    disp('Assuming order: p200; p020; p002; p011; p101; p110')
    [bary, cw, bernst, cbernst] = bezierTriangleWeights(nsubdiv, 2);
    control_pts = in_pts;
    
    pos2d_control_pts = canonical_pos;
    pos2d_control_pts(4, :) = 0.5 * (canonical_pos(2, :) + canonical_pos(3, :));
    pos2d_control_pts(5, :) = 0.5 * (canonical_pos(1, :) + canonical_pos(3, :));
    pos2d_control_pts(6, :) = 0.5 * (canonical_pos(1, :) + canonical_pos(2, :));
else
    disp('Rendering cubic bezier triangle')
    [bary, cw, bernst, cbernst] = bezierTriangleWeights(nsubdiv, 3);
    
    if size(in_pts, 1) == 10
        disp('Rendering color sail with preset control points')
        disp('Assuming order: p300; p030; p003; p012; p021; p102; p201; p120; p210; p111')
        control_pts = in_pts;
    elseif size(in_pts, 1) == 3
        disp('Rendering standard color sail')
        [control_pts, ~] = getSailControlPoints(in_pts, wind(1), wind(2), wind(3));
    end
    
    [~, pos2d_control_pts] = getSailControlPoints(canonical_pos, 0.0, 1/3.0, 1/3.0);
end
control_pts = reshape(control_pts, [1, size(control_pts, 1), size(control_pts, 2)]);
pos2d_control_pts = reshape(pos2d_control_pts, [1, size(pos2d_control_pts, 1), size(pos2d_control_pts, 2)]);
colors_out = squeeze(sum(bernst .* control_pts, 2));  

% Get triangle locations for 3D visualization
tri_pts = squeeze(sum(cbernst .* control_pts, 2));

% Get triangle locations for 2D visualization
tri2d_pts = squeeze(sum(cbernst .* pos2d_control_pts, 2));

% Plot 3D triangle
%figure;
plotTris(tri_pts, colors_out, true);
xlabel('R')
ylabel('G')
zlabel('B')
set(gca,'xticklabel',[])
set(gca,'yticklabel',[])
set(gca,'zticklabel',[])
hold on
cp = squeeze(control_pts);
%scatter3(cp(:, 1), cp(:, 2), cp(:, 3), 20, [0, 0, 0], 'filled');

% Plot 2D triangle
figure;
plotTris(tri2d_pts, colors_out, false);
pbaspect([1,1,1])
view(0,90)
grid off
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])
set(gca,'visible','off')

control_pts = squeeze(control_pts);
end

function plotTris(v, colors, use_line)
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
   if use_line
       fill3(X(:, i), Y(:, i), Z(:, i), colors(i, :)); %, 'LineStyle','none');
   else
       fill3(X(:, i), Y(:, i), Z(:, i), colors(i, :), 'LineStyle','none');
   end
   hold on;
end
xlim([0, 1.0]);
ylim([0, 1.0]);
zlim([0, 1.0]);
grid on
end

