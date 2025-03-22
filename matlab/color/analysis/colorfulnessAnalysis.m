FA = dlmread('../../../experiments/color/misc/colorfulness/fine_art_color.txt');
GD = dlmread('../../../experiments/color/misc/colorfulness/graphic_design_color.txt');
VIZ = dlmread('../../../experiments/color/misc/colorfulness/viz_color.txt');
PH = dlmread('../../../experiments/color/misc/colorfulness/photo_color.txt');
GAP = dlmread('../../../experiments/color/misc/colorfulness/gap_color.txt');


FA(:,2) = 0;
GD(:,2) = 1;
VIZ(:,2) = 2;
PH(:,2)=3;
GAP(:,2)=4;
all = [FA;GD;VIZ;PH;GAP];

figure; 
boxplot(all(:,1), all(:,2), 'Labels', {'art', 'gf.des.', 'viz', 'photo', 'GAP'})

edges = [0, 20, 40, 60, 80, 100, 120, 140, 200];
%edges = [0, 40, 80, 120, 200];
hFA = histc(FA(:,1), edges);
hGD = histc(GD(:,1), edges);
hVIZ = histc(VIZ(:,1), edges);
hPH = histc(PH(:,1), edges);
hGAP = histc(GAP(:,1), edges);
H = hFA / sum(hFA);
H(:, 2) = hGD / sum(hGD);
H(:, 3) = hVIZ / sum(hVIZ);
H(:, 4) = hPH / sum(hPH);
H(:, 5) = hGAP / sum(hGAP);

Hclean = histc([FA(:,1); GD(:,1); VIZ(:,1); PH(:,1)], edges);
H = Hclean/sum(Hclean);
hGAP = histc(GAP(:,1), edges);
H(:,2) = hGAP / sum(hGAP);

figure();
hold on
name = {'<20';'<40';'<60';'<80';'<100'; '<120'; '<140'; '>140'};
base_color = rgb2hsv([253	105	34	]/255.0)
base_color(2) = 1.0;
for d = 1:2
    for i = length(hFA):-1:1
        xval = 220*(d-1) + edges(i) + 10;
        ncolor = base_color;
        ncolor(2) = ncolor(2) * i / numel(hFA);
        b = bar(xval, H(i, d), 20);
        set(b,'FaceColor', hsv2rgb(ncolor));
    end
end
%set(gca,'xticklabel',name)