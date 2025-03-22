function [ colorfulness ] = batchComputeColorfulness(inputTextFile, dataDir)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

fid = fopen(inputTextFile);
txt = textscan(fid,'%s','delimiter','\n');

colorfulness = zeros(size(txt{1},1),1);
for i=1:size(txt{1},1)
    try
        colorfulness(i,1) = getColourfulness(imread(fullfile(dataDir, txt{1}{i})));
    catch ME
        disp('Err')
    end
end

end