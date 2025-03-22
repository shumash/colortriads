function V = sampleImage(imageFile, nsamples, graph)
%SAMPLEIMAGE Samples colors from an image file, and optionally graphs them
%   imageFile - string with the filename; else image itself
%   nsamples - number of samples to generate
%   graph - bool indicating whether or not to graph the samples
% returns matrix with one color per row, with cols 1-3 as rgb (0.0-1.0), and 4-6
%         as LAB

if isa(imageFile, 'char')
    [~,name,ext] = fileparts(imageFile);
    if strcmp(ext, '.png') || strcmp(ext, '.PNG')
        [A, ~, At] = imread(imageFile);
        I = A;
        if size(At, 1) == size(A, 1)
            I(:, :, 4) = At;
        end
    else
        A = imread(imageFile);
        I = A;
    end
else
    name = 'image';
    I = imageFile;
end

if graph
    figure; title(name)
    %ax = subplot(2,1,1);
    %subimage(I(:,:,1:3)); 
end
sz = size(I);
nchannels = sz(3);
R = reshape(im2double(I), [sz(1) * sz(2), nchannels]);
if nchannels > 3
    R = R(R(:,4) > 0.95, :);
end

sz = size(R);
r = randperm(sz(1), nsamples);
V = R(r, 1:3);
V(:,4:6) = rgb2lab(V);

if graph
    V(:,4) = 1;
    % ax = subplot(2,1,2);
    ax = gca;
    scatter3(ax, V(1:end,1), V(1:end,2), V(1:end,3), 40, V(1:end,1:3), 'filled'); %title('RGB')
    % set(fh, 'Position', [100, 100, 1200, 1200]);  % Uncomment for consistent scale
    
    ax.GridAlpha = 0.4;
    ax.GridLineStyle = ':';
    ax.XLabel.String = 'R';
    ax.YLabel.String = 'G';
    ax.ZLabel.String = 'B';
    ax.XLim = [ 0, 1.0]
    ax.YLim = [ 0, 1.0]
    ax.ZLim = [ 0, 1.0]
    ax.LineWidth = 1.5;
    
    % plotRGB(V); title(name);
    V(:,4:6) = rgb2lab(V(:,1:3));
    figure; scatter3(V(1:end,4), V(1:end,5), V(1:end,6), 40, V(1:end,1:3), 'filled'); title('LAB')
end

end
