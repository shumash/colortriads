function [tset_results, tsets] = procTemporalLogs2()

RUN_DIR='/Users/shumash/Documents/Coding/Animation/animation/experiments/color/runs/';

D0 = fullfile(RUN_DIR, 'palette_graphs4_reg', 'noreg__rw_gr0w3_img512rgb_pwidth32_colors3_subdiv6', 'proc_logs');
D1 = fullfile(RUN_DIR, 'palette_graphs4_reg', 'reg0.1_rw_gr0w3_img512rgb_pwidth32_colors3_subdiv6', 'proc_logs');
D2 = fullfile(RUN_DIR, 'PALETTE_patches_wind', 'w3p.1_rw_gr0w3_img512rgb_pwidth32_colors3_subdiv6', 'proc_logs');
D3 = fullfile(RUN_DIR, 'PALETTE_patches_wind', 'w0p.1_rw_gr0w0_img512rgb_pwidth32_colors3_subdiv6', 'proc_logs');


tsets = {'target_hard', 'target_medium', 'target_easy'};
losses = {'L2RGB_10', 'KL_PAL_10', 'RECON_PERCENT_10'}; % recon percent only for wind/nowind
loss_idx = 2;

tset_results = {}
for t=1:length(tsets)
    fname = strcat('temporal2/', tsets{t}, '_patches_', losses{loss_idx}, '_temporal_mat.txt');
    tset_results{t}{1} = readSeq(fullfile(D0, fname));
    tset_results{t}{2} = readSeq(fullfile(D1, fname));
    tset_results{t}{3} = readSeq(fullfile(D2, fname));
    tset_results{t}{3} = readSeq(fullfile(D3, fname));
end


for t=1:length(tsets)
    h = plotCurves(tset_results{t}{1}, tset_results{t}{2}, tset_results{t}{3});
    legend(h, 'noreg', 'reg', 'preg');
    title(strcat(losses{loss_idx}, ' ', tsets{t}));
end


end

function h = plotCurves(v0, v1, v2)
maxep = 400;
figure();
X=[3:maxep]*200;
p1 = plot(X, v0(3:maxep), 'red');
hold on;
p2 = plot(X, v1(3:maxep), 'blue');
p3 = plot(X, v2(3:maxep), 'green');
%ylim([0 20]);
h = [p1;p2;p3];
end

function [E0, E1, E2] = readAll(dir0, dir1, dir2, fname)
    E0 = readSeq(fullfile(dir0, fname));
    E1 = readSeq(fullfile(dir1, fname));
    E2 = readSeq(fullfile(dir2, fname));
end
    

function Y = readSeq(f)
    T = dlmread(f);
    Y = T(2:end, :) * T(1, :)' ./ sum(T(1, :));
end