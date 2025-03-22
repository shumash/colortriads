function [colors, wind, nsubdiv] = colorSailFromFile(fname)
f = fopen(fname, 'r');
raw = fscanf(f, '%f');
colors = reshape(raw(1:9), [3,3]);
colors = colors';
wind = raw(10:12);
nsubdiv = raw(13);
fclose(f);
end

