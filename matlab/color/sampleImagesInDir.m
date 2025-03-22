function sampleImagesInDir(dir_name, nsamples, save_files)
%SAMPLEIMAGESINDIR Samples images in dir, and outputs samples as files in a subdir
%   See sampleImage for the format of the samples output for each image

files = dir(strcat(dir_name, '/', '*.jpg'));
out_dir = strcat(dir_name, '/', 'samples');
if save_files
    mkdir(out_dir)
end

total = 0;
for file = files'
    samples = sampleImage(fullfile(dir_name, file.name), nsamples, ~save_files);
    if save_files
        [~, name, ~] = fileparts(file.name);
        out_file = fullfile(out_dir, [name, '_samples.txt']);
        dlmwrite(out_file, samples, 'delimiter', ' ');
    end
    total = total + 1;
    if total > 10 && ~save_files
        break
    end
end

end
