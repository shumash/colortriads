function [All, datatags] = procPerformanceLogs()
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
RUN_DIR='/Users/shumash/Documents/Coding/Animation/animation/experiments/color/runs/';

Dbase = fullfile(RUN_DIR, 'palette_graphs4', 'nogap_rw_gr0w3_img512rgb_pwidth32_colors3_subdiv6', 'proc_logs');
D2 = fullfile(RUN_DIR, 'palette_graphs4', 'pg_gr0w3_img512rgb_pwidth32_colors3_subdiv6', 'proc_logs');
D3 = fullfile(RUN_DIR, 'palette_graphs4', 'pg_rw_gr0w3_img512rgb_pwidth32_colors3_subdiv6', 'proc_logs');

All = [];
loss_idx=2;
tsets = {'pretty', 'hard', 'medium', 'easy'}; %_patches', 'target_easy_patches', 'target_hard_patches', 'target_medium_patches' };

models = {Dbase, D2, D3 };
model_tags = { 'nogap', 'mix', 'rw' };
tsets = {'fine_art', 'graphic_design', 'viz', 'photo','target_easy', 'target_medium', 'target_hard', 'GAP_easy', 'GAP_medium',  'GAP_hard'};
datatags = {};
for k=1:length(tsets)
    for j=1:length(models)
        f = fullfile(models{j}, 'perform', strcat(tsets{k}, '_patches_mat.txt'));
        disp(f);
        P0 = dlmread(f);
        %P0 = dlmread(fullfile(Dbase, 'perform', strcat(tsets{k}, '_mat.txt')));
        %P2 = dlmread(fullfile(D2, 'perform', strcat(tsets{k}, '_mat.txt')));
        %P3 = dlmread(fullfile(D3, 'perform', strcat(tsets{k}, '_mat.txt')));
        
        %block = [P0(loss_idx, :)', P2(loss_idx, :)', P3(loss_idx, :)', ones([size(P0,2), 1]).*k];
        idx = ((k-1)*length(models)+j);
        block = [P0(loss_idx, :)', ones([size(P0,2), 1]).*idx];
        datatags{idx} = strcat(model_tags{j}, '+', tsets{k});
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

end

