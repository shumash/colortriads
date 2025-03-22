function [colors, wind, sail_colors, meta] = colorSailOptimizeFromScratch(target_colors, opts, nsubdiv)

start_colors = colorSailGuessCornerColors(target_colors);
start_wind = [ 0.0, 1/3.0, 1/3.0 ];

[colors, wind, sail_colors, meta] = colorSailOptimize(start_colors, start_wind, target_colors, opts, nsubdiv);
end


