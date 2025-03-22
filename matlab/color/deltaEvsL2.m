% [ img, lab_out, rgb_out, dist_out ] = sampleLabDist(10, 2000);
% 
% deltaE = sRGB2CIEDeltaE(squeeze(rgb_out(:,1,:)), squeeze(rgb_out(:,2,:)), 'cie00');
% figure;
% plot(deltaE, dist_out, '*');
% xlabel('deltaE');
% ylabel('L2 in LAB');
% title('cie00');

I0 = im2double(imread('/tmp/patches/patch0.png'));
I1 = im2double(imread('/tmp/patches/patch1.png'));

R = I0(:,:,1);
RGB0 = R(:);
G = I0(:,:,2);
RGB0(:,2) = G(:);
B = I0(:,:,3);
RGB0(:,3) = B(:);

R = I1(:,:,1);
RGB1 = R(:);
G = I1(:,:,2);
RGB1(:,2) = G(:);
B = I1(:,:,3);
RGB1(:,3) = B(:);


D = sqrt(squeeze(sum((rgb2lab(RGB0) - rgb2lab(RGB1)).^2, 2)));