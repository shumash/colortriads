function [All, datatags, stats] = procPerformanceLogs2()
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
RUN_DIR='/Users/shumash/Documents/Coding/Animation/animation/experiments/color/runs/';

D0 = fullfile(RUN_DIR, 'palette_graphs4_reg', 'noreg__rw_gr0w3_img512rgb_pwidth32_colors3_subdiv6', 'proc_logs');
D1 = fullfile(RUN_DIR, 'palette_graphs4_reg', 'reg0.1_rw_gr0w3_img512rgb_pwidth32_colors3_subdiv6', 'proc_logs');
D2 = fullfile(RUN_DIR, 'PALETTE_patches_wind', 'w3p.1_rw_gr0w3_img512rgb_pwidth32_colors3_subdiv6', 'proc_logs');
D3 = fullfile(RUN_DIR, 'PALETTE_patches_wind', 'w0p.1_rw_gr0w0_img512rgb_pwidth32_colors3_subdiv6', 'proc_logs');

All = [];
losstags = { 'L2RGB', 'KL', 'E%' };
loss_idx=3;
% 1, 2, 3
% L2RGB_1:0 KL_PAL_1:0 RECON_PERCENT_1:0 RECON_PERCENT_3:0 total_loss:0
tsets = {'pretty', 'hard', 'medium', 'easy'}; %_patches', 'target_easy_patches', 'target_hard_patches', 'target_medium_patches' };

models = {D2, D3}; %{D0, D1, D2 };
model_tags = {'WIND', 'nowind'}; %{ 'noreg', 'KL', 'patchKL' };
tsets = {'fine_art', 'graphic_design', 'viz', 'photo','target_easy', 'target_medium', 'target_hard', 'GAP_easy', 'GAP_medium',  'GAP_hard'};
datatags = {};

prc = [5, 10, 20, 30];
stats = [];  % mean, median, std, prctiles
for k=1:length(tsets)
    for j=1:length(models)
        f = fullfile(models{j}, 'perform2', strcat(tsets{k}, '_patches_mat.txt'));
        disp(f);
        P0 = dlmread(f, ' ', 1, 2);
        values = P0(:,loss_idx);
        srow = [mean(values(:)), median(values(:)), std(values(:)), prctile(values(:), prc)];
        stats = [stats; srow];
        
        %block = [P0(loss_idx, :)', P2(loss_idx, :)', P3(loss_idx, :)', ones([size(P0,2), 1]).*k];
        idx = ((k-1)*length(models)+j);
        block = [P0(:,loss_idx), ones([size(P0,1), 1]).*idx];
        
        infostr = strcat(model_tags{j}, '+', tsets{k});
        datatags{idx} = infostr;
        if size(All, 1) == 0
            All = block;
        else
            All = [All; block];
        end
    end
end

figure; boxplot(All(:,1), All(:,2), 'Labels', datatags);
set(gca,'XTickLabelRotation',45);
set(findobj(gca,'type','line'),'linew',1.2);
title(losstags{loss_idx});

end

