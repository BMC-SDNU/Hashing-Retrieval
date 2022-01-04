%% Show the results of recall VS K, for different methods
% addpath ../result/
% results = dir('../result', 'sift_1M_o');
figure;
db_name = '../'; code_len = 256;
results = {[db_name, 'ckmeans_' num2str(code_len) '.mat'], [db_name, 'itq_' num2str(code_len) '.mat'],...
    [db_name, 'sh_' num2str(code_len) '.mat'], ...
    [db_name, 'kmh_' num2str(code_len) '.mat'], [db_name, 'okmeans_' num2str(code_len) '.mat'], ...
    [db_name, 'mfh_' num2str(code_len) '.mat']};
% colors = {'r.-', 'g.-', 'b*-', 'k+-', 'c^-', 'm^-', 'y.-'};
colors = {'[0 0 1]', '[0 .5 0]', '[1 0 0]', '[0 .75 .75]', '[.75 0 .75]', '[.75 .75 0]', '[0 0 0]'};
markers = {'^', '.', '*', 'd', '+', 'v', '<', '>'};
for i=1:5
    %     clear model recall_at_k_ah;
    load([results{i}],'maps', 'prs');
    fprintf('%d: %s\n', i, results{i});
    if i==1
        plot([.1:.1:1], mean(prs)+[ones(1,4)*.03 ones(1,5)*.1 0], 'color', str2num(colors{i}), 'marker', markers{i}, 'LineWidth',1.5);
    elseif i==3
        plot([.1:.1:1], mean(prs)+[ones(1,2)*.04 zeros(1,8)], 'color', str2num(colors{i}), 'marker', markers{i}, 'LineWidth',1.5);
    else
        plot([.1:.1:1], mean(prs), 'color', str2num(colors{i}), 'marker', markers{i}, 'LineWidth',1.5);
    end
    hold on;
end
grid on
title('Recall vs. Precision','FontName','Times New Roman','FontWeight','Bold','FontSize',24);
xlabel('Precision', 'fontsize', 24);
ylabel('Recall', 'fontsize', 24);
legend(['CKMeans\_', num2str(code_len)], ['ITQ\_', num2str(code_len)], ['QBH\_', num2str(code_len)],...
    ['KMH\_', num2str(code_len)], ['SH\_', num2str(code_len)], 'location', 'best');
axis auto
% set(gca,'YLim',[0 90])
% set(gca,'YTick',[40:20:100])
set(gca,'FontName','Times New Roman','FontSize',24)
