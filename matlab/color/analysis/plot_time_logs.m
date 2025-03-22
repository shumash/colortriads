function [ d_easy, d_medium, d_hard ] = plot_time_logs(odir, epoch)
%First run: ./scripts/color/analysis/process_log_losses.sh RAW_LOG.txt ODIR
%Then run this with ODIR as input


d_easy = process_counts(dlmread(fullfile(odir, 'easy_raw.txt')));
d_medium = process_counts(dlmread(fullfile(odir, 'medium_raw.txt')));
d_hard = process_counts(dlmread(fullfile(odir, 'hard_raw.txt')));

x=[1:numel(d_hard)] * epoch;

figure;
plot(x, d_easy, ':', 'color', [42,	199,	98]/255.0	, 'LineWidth', 2);
hold on
plot(x, d_medium, 'color', [253	186	75	]/255.0, 'LineWidth',2);
hold on
plot(x, d_hard, '--', 'color', [252	77	130	] /255.0, 'LineWidth',2);
legend('easy','medium', 'hard')
set(gca,'FontSize',15)

fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 6 3];
%print('/tmp/5by3DimensionsFigure.png','-dpng','-r0')

end


function normdata = process_counts(rdata)
    counts = rdata(1,:);
    losses = rdata(2:end,:);
    total = sum(counts);
    
    normdata = (losses * counts') / total;
end