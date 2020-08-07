%%% Histogram of DBSI and DTI metrics on Gd-T1W hyper-intensity regions, Gd-T1W hypo-intnesity regions and FLAIR hyper-intensity regions 
clear all;
maps = {'dti_adc_map', 'dti_fa_map', 'iso_adc_map', 'hindered_ratio_map', 'restricted_ratio_2_map', 'fiber_ratio_map'};
%ROIs = {'Gd_hyper', 'Gd_hypo', 'FLAIR_Gd'};
ROIs = {'Gd_hyper', 'FLAIR_Gd'};
dirDBSI = ['C:\Users\zye01\Desktop\GBM\GBM_3\DBSI_results_0.2_0.2_1.5_1.5_2.5_2.5\'];
dirROI =  ['C:\Users\zye01\Desktop\GBM\GBM_3\'];
savePath = ['C:\Users\zye01\Desktop\GBM\GBM_3\data'];
if ~exist(savePath, 'dir')
  mkdir(savePath);
end;

for i = 1:numel(maps);
    map = maps{i};
    img = char(map);
    filename_MRI = fullfile(dirDBSI,[img, '.nii']);
    data_MRI = load_nii(filename_MRI);
    data_MRI = data_MRI.img;
    for j = 1:numel(ROIs);
        ROI = ROIs{j};
        roi = char(ROI);
        filename_ROI = fullfile(dirROI,[roi, '.nii']);
        data_ROI = load_nii(filename_ROI);
        data_ROI = data_ROI.img;
        data = double(data_ROI).*double(data_MRI);
        index = find(data > 0);
        data = data(index);
        file_name = [img, '_', roi, '_data.mat'];
        save([savePath, '/', file_name]);
    end;
end;


%%%% Gd enhancing lesion 
subplot(1, 3, 1);

load(fullfile(savePath, 'hindered_ratio_map_Gd_hyper_data'), 'data');
x2 = data;
h2 = histogram(x2, 'Normalization', 'pdf', 'facecolor', 'blue');
h2.NumBins = 30;
hold on;
load(fullfile(savePath, 'fiber_ratio_map_Gd_hyper_data'), 'data');
x3 = data;
h3 = histogram(x3, 'Normalization', 'pdf', 'facecolor', [0 .5 0]);
h3.NumBins = 15;
hold on;
load(fullfile(savePath, 'restricted_ratio_2_map_Gd_hyper_data'), 'data');
x1 = data;
h1 = histogram(x1, 'Normalization', 'pdf', 'facecolor', 'red');
h1.NumBins = 30;
hold on;

[f,x2] = ksdensity(x2, 'Bandwidth', 0.02);
plot(x2, f, 'LineWidth', 3, 'color', 'blue');
hold on;
[f,x3] = ksdensity(x3, 'Bandwidth', 0.02);
plot(x3, f, 'LineWidth', 3, 'color', 'green');
hold on;
[f,x1] = ksdensity(x1, 'Bandwidth', 0.02);
plot(x1, f, 'LineWidth', 3, 'color', 'red');
hold on;

axis([0 1 0 6]);
ax = gca;
ax.XTick = [0 0.5 1];
ax.YTick = [0 3 6];
ax.TickDir = 'out';
ax.TickDir = 'out';
%title('Gd enhancing lesion');
set(gca, 'fontsize', 20, 'Fontweight', 'bold', 'linewidth', 2);
box off;
hold on;


% %%%% Gd non-enhancing areas
% subplot(1, 3, 2);
% %plot histogram;
% load(fullfile(savePath, 'hindered_ratio_map_Gd_hypo_data'), 'data');
% x3 = data;
% h3 = histogram(x3, 'Normalization', 'pdf', 'facecolor', 'blue');
% h3.NumBins = 30;
% hold on;
% load(fullfile(savePath,'fiber_ratio_map_Gd_hypo_data'), 'data');
% x1 = data;
% h1 = histogram(x1, 'Normalization', 'pdf', 'facecolor', [0 .5 0]);
% h1.NumBins = 10;
% hold on;
% load(fullfile(savePath, 'restricted_ratio_2_map_Gd_hypo_data'), 'data');
% x2 = data;
% h2 = histogram(x2, 'Normalization', 'pdf', 'facecolor', 'red');
% h2.NumBins = 30;
% hold on;

% [f,x3] = ksdensity(x3,'Bandwidth',0.04);
% plot(x3, f, 'LineWidth', 3, 'color', 'blue');
% hold on;
% [f,x1] = ksdensity(x1, 'Bandwidth', 0.01);
% plot(x1, f, 'LineWidth', 3, 'color', 'green');
% hold on;
% [f,x2] = ksdensity(x2, 'Bandwidth', 0.05);
% plot(x2, f, 'LineWidth', 3, 'color', 'red');
% hold on;
% 
% axis([0 1 0 24]);
% ax = gca;
% ax.XTick = [0 0.5 1];
% ax.YTick = [0 12 24];
% ax.TickDir = 'out';
% ax.TickDir = 'out';
% %title('Gd non-ehancing area');
% set(gca, 'fontsize', 20, 'Fontweight', 'bold', 'linewidth', 2);
% box off;

%%%%%% FLAIR hyperintensity region
subplot(1, 3, 2);
load(fullfile(savePath, 'hindered_ratio_map_FLAIR_Gd_data'), 'data');
x2 = data;
h2 = histogram(x2, 'Normalization', 'pdf', 'facecolor', 'blue');
h2.NumBins = 30;
hold on;
load(fullfile(savePath, 'fiber_ratio_map_FLAIR_Gd_data'), 'data');
x3 = data;
h3 = histogram(x3, 'Normalization', 'pdf', 'facecolor', [0 .5 0]);
h3.NumBins = 30;
hold on;
load(fullfile(savePath, 'restricted_ratio_2_map_FLAIR_Gd_data'), 'data');
x1 = data;
h1 = histogram(x1, 'Normalization', 'pdf', 'facecolor', 'red');
h1.NumBins = 30;
hold on;


[f,x2] = ksdensity(x2, 'Bandwidth', 0.02);
plot(x2, f, 'LineWidth', 3, 'color', 'blue');
hold on;
[f,x3] = ksdensity(x3, 'Bandwidth', 0.02);
plot(x3, f, 'LineWidth', 3, 'color', 'green');
hold on;
[f,x1] = ksdensity(x1, 'Bandwidth', 0.02);
plot(x1, f, 'LineWidth' ,3, 'color', 'red');
hold on;

axis([0 1 0 4]);
ax = gca;
ax.XTick = [0 0.5 1];
ax.YTick = [0 2 4];
ax.TickDir = 'out';
ax.TickDir = 'out';
%title('FLAIR Lesion');
set(gca, 'fontsize', 20, 'Fontweight', 'bold', 'linewidth', 2);
box off;




