figure();
hold on;


P = stats(:, 4:7);
P(:,4) = P(:,4) - P(:,3);
P(:,3) = P(:,3) - P(:,2);
P(:,2) = P(:,2) - P(:,1);

C = [84	223	166	;
    196	235	111	;		%123	206	155	;
    253	189	80	; %175	213	127	;
   253	136	77	] / 255.0;
%253	126	73	] / 255.0;

% C = [105	230	108;
%     141	197	79	;
%     191	157	52	;
% 236	124	39	] / 255.0;
% [231	147	51; 
% 158	181	81	;
% 89	215	124	] /255.0;

%ds = [9, 11, 13];
ds = [1,3,5,7,21];
ntags = {};
for j = 1:length(ds)
    ntags{j} = datatags{ds(j)};
end

b = bar(P(ds,:), 'stacked');
b(1).Parent.Parent.Colormap = C;
legend(b, {'5th perc', '10th perc', '20th perc', '30th perc'});
set(gca, 'XTick', 1:length(ds));
set(gca, 'XTickLabel', ntags);


return;


figure();
D = stats;
hold on

l_stat_names = {'mean', 'median', 'std', 'perc5', 'perc10', 'perc20'};
statidx = 6;  % mean, meadian, std, perc5, perc10, perc15
for i = 1:length(datatags)
  xval = i;
  yval = D(i, statidx);
  color = [0.5, 0.5, 0.5];
  b = bar(xval, yval, 0.8);
  set(b,'FaceColor', color);
end

xlim([0,length(datatags)]);
set(gca, 'XTick', 1:length(datatags));
set(gca, 'XTickLabel', datatags);
set(gca,'XTickLabelRotation',45);
%set(findobj(gca,'type','line'),'linew',1.2);

title(l_stat_names{statidx});
return;

indices= [8,7,4,3,6,5,2,1];
nowind = [8, 4, 6, 2];
wind = [7, 3, 5, 1];

ntags = {};
for j = 1:length(nowind)
    ntags{j} = datatags{nowind(j)};
end

figure;
x = [1,2, 4,5, 7,8, 10,11];
xnw = x(1:2:end);
xw = x(2:2:end);
errorbar(xnw,D(nowind,1),D(nowind,3),'-s','MarkerSize',10,...
    'MarkerEdgeColor','red','MarkerFaceColor','red');
hold on;
errorbar(xw,D(wind,1),D(wind,3),'-s','MarkerSize',10,...
    'MarkerEdgeColor','blue','MarkerFaceColor','blue');

set(gca, 'XTick', [1.5, 4.5, 7.5, 10.5]);
set(gca, 'XTickLabel', ntags);