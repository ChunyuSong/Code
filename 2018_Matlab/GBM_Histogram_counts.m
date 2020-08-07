%%% Histogram of DBSI and DTI metrics on Gd-T1W hyper-intensity regions, Gd-T1W hypo-intnesity regions and FLAIR hyper-intensity regions 
clear all;
maps = {'dti_adc_map','dti_fa_map','iso_adc_map','hindered_ratio_map','restricted_ratio_2_map','fiber_ratio_map'};
%ROIs = {'Gd_hyper','Gd_hypo','FLAIR_Gd','NAWM'};
ROIs = {'Gd_hyper','Gd_hypo','FLAIR_Gd'};
%ROIs = {'Gd_hyper', 'FLAIR_Gd'};
dirDBSI = ['C:\Users\zye01\Desktop\GBM\GBM_1\DBSI_results_0.2_0.2_1.5_1.5_2.5_2.5\'];
dirROI =  ['C:\Users\zye01\Desktop\GBM\GBM_1\'];
savePath = ['C:\Users\zye01\Desktop\GBM\GBM_1\data'];
if ~exist(savePath, 'dir')
  mkdir(savePath);
end;

for i = 1:numel(maps);
    map = maps{i};
    img = char(map);
    filename_MRI = fullfile(dirDBSI,[img,'.nii']);
    data_MRI = load_nii(filename_MRI);
    data_MRI = data_MRI.img;
    for j = 1:numel(ROIs);
        ROI = ROIs{j};
        roi = char(ROI);
        filename_ROI = fullfile(dirROI,[roi,'.nii']);
        data_ROI = load_nii(filename_ROI);
        data_ROI = data_ROI.img;
        data = double(data_ROI).*double(data_MRI);
        index = find(data > 0);
        data = data(index);
        file_name = [img,'_',roi,'_data.mat'];
        save([savePath,'/',file_name]);
    end;
end;


% restricted fraction
clear all;
subplot(1,3,1);

load('C:\Users\zye01\Desktop\GBM\GBM_1\data\restricted_ratio_2_map_FLAIR_Gd_data', 'data');
x2 = data;
h2 = histogram(x2, 'facecolor', [0 .5 0]);
h2.NumBins = 30;
hold on;
load('C:\Users\zye01\Desktop\GBM\GBM_1\data\restricted_ratio_2_map_Gd_hyper_data', 'data');
x3 = data;
h3 = histogram(x3, 'facecolor', 'red');
h3.NumBins = 30;
hold on;
load('C:\Users\zye01\Desktop\GBM\GBM_1\data\restricted_ratio_2_map_Gd_hypo_data', 'data');
x1 = data;
h1 = histogram(x1, 'facecolor', 'blue');
h1.NumBins = 30;
hold on;
axis([0 1 0 15000]);
ax = gca;
ax.XTick = [0 0.5 1];
%ax.YTick = [0 4 8];
ax.TickDir = 'out';
ax.TickDir = 'out';
%title('Anisotropic Fraction');
set(gca,'fontsize',20,'Fontweight','bold','linewidth',2);
box off;


% hindered fraction
clear all;
subplot(1,3,2);
%plot histogram;

load('C:\Users\zye01\Desktop\GBM\GBM_1\data\hindered_ratio_map_FLAIR_Gd_data','data');
x2 = data;
h2 = histogram(x2, 'facecolor', [0 .5 0]);
h2.NumBins = 30;
hold on;
load('C:\Users\zye01\Desktop\GBM\GBM_1\data\hindered_ratio_map_Gd_hyper_data','data');
x3 = data;
h3 = histogram(x3, 'facecolor', 'red');
h3.NumBins = 30;
hold on;
load('C:\Users\zye01\Desktop\GBM\GBM_1\data\hindered_ratio_map_Gd_hypo_data','data');
x1 = data;
h1 = histogram(x1, 'facecolor', 'blue');
h1.NumBins = 30;
hold on;
axis([0 1 0 10000]);
ax = gca;
ax.XTick = [0 0.5 1];
%ax.YTick = [0 4 8];
ax.TickDir = 'out';
ax.TickDir = 'out';
%title('Anisotropic Fraction');
set(gca,'fontsize',20,'Fontweight','bold','linewidth',2);
box off;


% anisotropic fraction
clear all;
subplot(1,3,3);
load('C:\Users\zye01\Desktop\GBM\GBM_1\data\fiber_ratio_map_FLAIR_Gd_data','data');
x3 = data;
h3 = histogram(x3, 'facecolor', [0 .5 0]);
h3.NumBins = 30;
hold on;
load('C:\Users\zye01\Desktop\GBM\GBM_1\data\fiber_ratio_map_Gd_hyper_data','data');
x1 = data;
h1 = histogram(x1, 'facecolor', 'red');
h1.NumBins = 30;
hold on;
load('C:\Users\zye01\Desktop\GBM\GBM_1\data\fiber_ratio_map_Gd_hypo_data','data');
x2 = data;
h2 = histogram(x2,  'facecolor', 'blue');
h2.NumBins = 30;
hold on;
axis([0 1 0 10000]);
ax = gca;
ax.XTick = [0 0.5 1];
%ax.YTick = [0 4 8];
ax.TickDir = 'out';
ax.TickDir = 'out';
%title('Anisotropic Fraction');
set(gca,'fontsize',20,'Fontweight','bold','linewidth',2);
box off;


