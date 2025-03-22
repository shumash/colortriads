function procTemporalLogs()


RUN_DIR='/Users/shumash/Documents/Coding/Animation/animation/experiments/color/runs/';

D0 = fullfile(RUN_DIR, 'palette_graphs3_reg', 'noreg__rw_gr0w3_img512rgb_pwidth32_colors3_subdiv6', 'proc_logs');
%'palette_graphs4', 'nogap_rw_gr0w3_img512rgb_pwidth32_colors3_subdiv6', 'proc_logs');
D2 = fullfile(RUN_DIR, 'palette_graphs4_reg', 'reg0.1_rw_gr0w3_img512rgb_pwidth32_colors3_subdiv6', 'proc_logs');
D3 = fullfile(RUN_DIR, 'palette_graphs4_reg', 'patchreg_rw_gr0w3_img512rgb_pwidth32_colors3_subdiv6', 'proc_logs');
%pg_gr0w3_img512rgb_pwidth32_colors3_subdiv6', 'proc_logs');
%D3 = fullfile(RUN_DIR, 'palette_graphs4', 'pg_rw_gr0w3_img512rgb_pwidth32_colors3_subdiv6', 'proc_logs');

losses = { '
Feasy = 'temporal/easy_total_loss0_temporal_mat.txt';
Fmed = 'temporal/medium_total_loss0_temporal_mat.txt';
Fhard = 'temporal/hard_total_loss0_temporal_mat.txt';

[E0, E1, E2] = readAll(Dbase, D2, D3, Feasy);
[M0, M1, M2] = readAll(Dbase, D2, D3, Fmed);
[H0, H1, H2] = readAll(Dbase, D2, D3, Fhard);


plotCurves(H1, M1, E1);

plotCurves(E0, E1, E2);

end

function plotCurves(v0, v1, v2)
maxep = 400;
figure();
X=[3:maxep]*200;
plot(X, v0(3:maxep), 'red');
hold on;
plot(X, v1(3:maxep), 'blue');
plot(X, v2(3:maxep), 'green');
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