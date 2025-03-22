function makePrettySailGraph(inputImageFile, inputSailTextFile)

[colors, wind, nsubdiv] = colorSailFromFile(inputSailTextFile);
[h, hc] = computePatch3DHist(imread(inputImageFile), 512, 32, 10, true);
%figure; 
plot3DHistogram(h, hc); hold on;
bezierTriangleClean(colors, wind', nsubdiv);

end

