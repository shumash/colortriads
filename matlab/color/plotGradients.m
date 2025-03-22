gradDir = '../../data/mixing/physical/gradients/test_grad2_';
gradients = cellstr(['grad3_ac_tw_cb.jpg'; 'grad3_pr_tw_cb.jpg']);
 
            
figure
for i = 1:8 %numel(gradients)
    plotGradient(strcat(gradDir, num2str(i), '.jpg'), 3, false);
    hold on
end

title('Gradients');
