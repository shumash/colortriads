function colorSailToFile(fname, colors, wind, nsubdiv)
fileID = fopen(fname,'w');
nbytes = fprintf(fileID,'%0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %d\n', colors', wind, nsubdiv);
fclose(fileID);
end