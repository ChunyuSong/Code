% clear;
% 
% data_path = ['D:\2020_Data\'];
% 
% load(fullfile(data_path, 'PCa_grant_data.xlsx'));

A = xlsread('PCa_grant_data.xlsx', 'all');

x1 = A(1:104, 1);
x2 = A(1:104, 2);
x3 = A(1:104, 3);

x1_mean = mean(x1);
x1_median = median(x1);
x1_std = std(x1);

display(x1_mean);
display(x1_median);
display(x1_std);

x2_mean = mean(x2);
x2_median = median(x2);
x2_std = std(x2);

display(x2_mean);
display(x2_median);
display(x2_std);

x3_mean = mean(x3);
x3_median = median(x3);
x3_std = std(x3);

display(x3_mean);
display(x3_median);
display(x3_std);

% h1 = histogram(x1, 'Normalization', 'pdf', 'facecolor', 'red');
% h1.NumBins = 30;
% hold on;

% [f, x1] = ksdensity(x1, 'Bandwidth', 0.08);
% plot(x1, f, 'LineWidth', 3, 'color', 'red');
% hold on;

x_values = -0.1:0.001:0.6;

pd = fitdist(x1, 'Normal');
x1 = pdf(pd, x_values);
plot(x_values, x1, 'LineWidth', 3, 'color', 'red');
hold on;

pd = fitdist(x2, 'Normal');
x2 = pdf(pd, x_values);
plot(x_values, x2, 'LineWidth', 3, 'color', 'blue');

pd = fitdist(x3, 'Normal');
x3 = pdf(pd, x_values);
plot(x_values, x3, 'LineWidth', 3, 'color', 'green');

% h2 = histogram(x2, 'Normalization', 'pdf', 'facecolor', 'blue');
% h2.NumBins = 30;
% hold on;

% [f, x2] = ksdensity(x2, 'Bandwidth', 0.08);
% plot(x2, f, 'LineWidth', 3, 'color', 'blue');

axis([-0.1 0.7 0 6]);
ax = gca;
% ax.XTick = [-0.1 0 0.1 0.2 0.3 0.4 0.5 0.6];
%ax.YTick = [0 2 4];
ax.TickDir = 'out';
ax.TickDir = 'out';
%title('FLAIR Lesion');
set(gca, 'fontsize', 15, 'Fontweight', 'bold', 'linewidth', 2);
xlabel('Restricted Fraction', 'Fontsize', 15) 
ylabel('Frequency', 'Fontsize', 15) 
legend({'Siemens Scan #1', 'Siemens Scan #2', 'GE Scan'}, 'Fontsize', 12, 'Location', 'northeast')
box off;