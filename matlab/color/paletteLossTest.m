function [err, errM, av, errF] = paletteLossTest(err_file)
A = ones(3)/8.0;
A(2,2) = 0;
err = im2double(rgb2gray(imread(err_file)));
errM = 10 .* err .^ 0.1 ./ (1 + exp(-0.5 * (err * 256 - 6.5)));
av = conv2(errM, A, 'same');

errF = av.* errM;

end