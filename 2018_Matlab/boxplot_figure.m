% %% Histogram of DBSI and DTI metrics on Gd-T1W hyper-intensity regions, Gd-T1W hypo-intnesity regions and FLAIR hyper-intensity regions 
% clear all;
% maps = {'dti_adc_map','dti_fa_map','iso_adc_map','hindered_ratio_map','restricted_ratio_2_map','fiber_ratio_map'};
% ROIs = {'ROI_Gd_hyper','ROI_Gd_hypo','ROI_FLAIR_Gd'};
% dirDBSI = ['C:\Users\zye01\Desktop\GBM\DBSI_results_0.2_0.2_1.5_1.5_2.5_2.5\'];
% dirROI =  ['C:\Users\zye01\Desktop\GBM\'];
% savePath = ['C:\Users\zye01\Desktop\GBM\data'];
% if ~exist(savePath, 'dir')
%   mkdir(savePath);
% end;
% 
% for i = 1:numel(maps);
%     map = maps{i};
%     img = char(map);
%     filename_MRI = fullfile(dirDBSI,[img,'.nii']);
%     data_MRI = load_nii(filename_MRI);
%     data_MRI = data_MRI.img;
%     for j = 1:numel(ROIs);
%         ROI = ROIs{j};
%         roi = char(ROI);
%         filename_ROI = fullfile(dirROI,[roi,'.nii']);
%         data_ROI = load_nii(filename_ROI);
%         data_ROI = data_ROI.img;
%         data = double(data_ROI).*double(data_MRI);
%         index = find(data > 0);
%         data = data(index);
%         subplot(2,3,i);
%         boxplot(data,j*ones(size(data)));
%         ax.TickDir = 'out';
%         ax.TickDir = 'out';
%         title(img);
%         set(gca,'fontsize',20,'Fontweight','bold','linewidth',2);
%         box off;
%     end;
% end;

%%
% %% DTI_ADC
% 
subplot(2,3,1);
clear all;
load('C:\Users\zye01\Desktop\GBM\data\dti_adc_map_ROI_Gd_hypo_data','data');
x1 = data;
load('C:\Users\zye01\Desktop\GBM\data\dti_adc_map_ROI_FLAIR_Gd_data','data');
x2 = data;
load('C:\Users\zye01\Desktop\GBM\data\dti_adc_map_ROI_Gd_hyper_data','data');
x3 = data;
boxplot([x1;x2;x3],[ones(size(x1));2*ones(size(x2));3*ones(size(x3))]);

subplot(2,3,2);
clear all;
load('C:\Users\zye01\Desktop\GBM\data\dti_fa_map_ROI_Gd_hypo_data','data');
x1 = data;
load('C:\Users\zye01\Desktop\GBM\data\dti_fa_map_ROI_FLAIR_Gd_data','data');
x2 = data;
load('C:\Users\zye01\Desktop\GBM\data\dti_fa_map_ROI_Gd_hyper_data','data');
x3 = data;
boxplot([x1;x2;x3],[ones(size(x1));2*ones(size(x2));3*ones(size(x3))]);

subplot(2,3,3);
clear all;
load('C:\Users\zye01\Desktop\GBM\data\iso_adc_map_ROI_Gd_hypo_data','data');
x1 = data;
load('C:\Users\zye01\Desktop\GBM\data\iso_adc_map_ROI_FLAIR_Gd_data','data');
x2 = data;
load('C:\Users\zye01\Desktop\GBM\data\iso_adc_map_ROI_Gd_hyper_data','data');
x3 = data;
boxplot([x1;x2;x3],[ones(size(x1));2*ones(size(x2));3*ones(size(x3))]);


subplot(2,3,4);
clear all;
load('C:\Users\zye01\Desktop\GBM\data\restricted_ratio_2_map_ROI_Gd_hypo_data','data');
x1 = data;
load('C:\Users\zye01\Desktop\GBM\data\restricted_ratio_2_map_ROI_FLAIR_Gd_data','data');
x2 = data;
load('C:\Users\zye01\Desktop\GBM\data\restricted_ratio_2_map_ROI_Gd_hyper_data','data');
x3 = data;
boxplot([x1;x2;x3],[ones(size(x1));2*ones(size(x2));3*ones(size(x3))]);


subplot(2,3,5);
clear all;
load('C:\Users\zye01\Desktop\GBM\data\hindered_ratio_map_ROI_Gd_hypo_data','data');
x1 = data;
load('C:\Users\zye01\Desktop\GBM\data\hindered_ratio_map_ROI_FLAIR_Gd_data','data');
x2 = data;
load('C:\Users\zye01\Desktop\GBM\data\hindered_ratio_map_ROI_Gd_hyper_data','data');
x3 = data;
boxplot([x1;x2;x3],[ones(size(x1));2*ones(size(x2));3*ones(size(x3))]);

subplot(2,3,6);
clear all;
load('C:\Users\zye01\Desktop\GBM\data\fiber_ratio_map_ROI_Gd_hypo_data','data');
x1 = data;
load('C:\Users\zye01\Desktop\GBM\data\fiber_ratio_map_ROI_FLAIR_Gd_data','data');
x2 = data;
load('C:\Users\zye01\Desktop\GBM\data\fiber_ratio_map_ROI_Gd_hyper_data','data');
x3 = data;
boxplot([x1;x2;x3],[ones(size(x1));2*ones(size(x2));3*ones(size(x3))]);


