%%% Histogram of DBSI and DTI metrics on Gd-T1W hyper-intensity regions, Gd-T1W hypo-intnesity regions and FLAIR hyper-intensity regions 
clear all;
maps = {'dti_adc_map','dti_fa_map','iso_adc_map','hindered_ratio_map','restricted_ratio_2_map','fiber_ratio_map'};
%ROIs = {'Gd_hyper','Gd_hypo','FLAIR_Gd','NAWM'};
ROIs = {'Gd_hyper','Gd_hypo','FLAIR_Gd'};
%ROIs = {'Gd_hyper', 'FLAIR_Gd'};
dirDBSI = ['C:\Users\zye01\Desktop\GBM\GBM_2\DBSI_results_0.2_0.2_1.5_1.5_2.5_2.5\'];
dirROI =  ['C:\Users\zye01\Desktop\GBM\GBM_2\'];
savePath = ['C:\Users\zye01\Desktop\GBM\GBM_2\data'];
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
%plot histogram;
load('C:\Users\zye01\Desktop\GBM\GBM_2\data\restricted_ratio_2_map_Gd_hypo_data', 'data');
x1 = data;
%h1 = histogram(x1,'Normalization','pdf','facecolor','blue');
h1 = histogram(x1, 'facecolor', 'blue');
h1.NumBins = 30;
%h1.BinWidth = 0.02;
hold on;
load('C:\Users\zye01\Desktop\GBM\GBM_2\data\restricted_ratio_2_map_FLAIR_Gd_data', 'data');
x2 = data;
%h2 = histogram(x2,'Normalization','pdf','facecolor',[0 .5 0]);
h2 = histogram(x2, 'facecolor', [0 .5 0]);
h2.NumBins = 30;
%h1.BinWidth = 0.08;
hold on;
load('C:\Users\zye01\Desktop\GBM\GBM_2\data\restricted_ratio_2_map_Gd_hyper_data', 'data');
x3 = data;
%h3 = histogram(x3,'Normalization','pdf','facecolor','red');
h3 = histogram(x3, 'facecolor', 'red');
h3.NumBins = 30;
%h1.BinWidth = 0.02;
hold on;
% load('C:\Users\zye01\Desktop\GBM\data\restricted_ratio_2_map_ROI_NAWM_data','data');
% x4 = data;
% h4 = histogram(x4,'Normalization','pdf','facecolor','yellow');
% h4.NumBins = 20;
% %h1.BinWidth = 0.02
% hold on;

%plot curve fitting
[f,x1] = ksdensity(x1,'Bandwidth',0.04);
plot(x1,f,'LineWidth',3,'color','blue');
hold on;
[f,x2] = ksdensity(x2,'Bandwidth',0.015);
plot(x2,f,'LineWidth',3,'color','green');
hold on;
[f,x3] = ksdensity(x3,'Bandwidth',0.04);
plot(x3,f,'LineWidth',3,'color','red');
hold on;
% [f,x4] = ksdensity(x4,'Bandwidth',0.05);
% plot(x4,f,'LineWidth',3,'color','yellow');
% hold on;

axis([0 1 0 8]);
ax = gca;
ax.XTick = [0 0.5 1];
ax.YTick = [0 4 8];
ax.TickDir = 'out';
ax.TickDir = 'out';
%title('Restricted Fraction');
set(gca,'fontsize',20,'Fontweight','bold','linewidth',2);
box off;
hold on;

% hindered fraction
clear all;
subplot(1,3,2);
%plot histogram;
load('C:\Users\zye01\Desktop\GBM\GBM_2\data\hindered_ratio_map_Gd_hypo_data','data');
x1 = data;
%h1 = histogram(x1,'Normalization','pdf','facecolor','blue');
h1 = histogram(x1, 'facecolor', 'blue');
h1.NumBins = 30;
%h1.BinWidth = 0.02;
hold on;
load('C:\Users\zye01\Desktop\GBM\GBM_2\data\hindered_ratio_map_FLAIR_Gd_data','data');
x2 = data;
%h2 = histogram(x2,'Normalization','pdf','facecolor',[0 .5 0]);
h2 = histogram(x2, 'facecolor', [0 .5 0]);
h2.NumBins = 30;
%h1.BinWidth = 0.08
hold on;
load('C:\Users\zye01\Desktop\GBM\GBM_2\data\hindered_ratio_map_Gd_hyper_data','data');
x3 = data;
%h3 = histogram(x3,'Normalization','pdf','facecolor','red');
h3 = histogram(x3, 'facecolor', 'red');
h3.NumBins = 30;
%h1.BinWidth = 0.02
hold on;
% load('C:\Users\zye01\Desktop\GBM\data\hindered_ratio_map_ROI_NAWM_data','data');
% x4 = data;
% h4 = histogram(x4,'Normalization','pdf','facecolor','yellow');
% h4.NumBins = 20;
% %h1.BinWidth = 0.02
% hold on;

%plot curve fitting
[f,x1] = ksdensity(x1,'Bandwidth',0.02);
plot(x1,f,'LineWidth',3,'color','blue');
hold on;
[f,x2] = ksdensity(x2,'Bandwidth',0.02);
plot(x2,f,'LineWidth',3,'color','green');
hold on;
[f,x3] = ksdensity(x3,'Bandwidth',0.04);
plot(x3,f,'LineWidth',3,'color','red');
hold on;
% [f,x4] = ksdensity(x4,'Bandwidth',0.05);
% plot(x4,f,'LineWidth',3,'color','yellow');
% hold on;

axis([0 1 0 5]);
ax = gca;
ax.XTick = [0 0.5 1];
ax.YTick = [0 2.5 5];
ax.TickDir = 'out';
ax.TickDir = 'out';
%title('Hindered Fraction');
set(gca,'fontsize',20,'Fontweight','bold','linewidth',2);
box off;
hold on;

% anisotropic fraction
clear all;
subplot(1,3,3);
%plot histogram;
load('C:\Users\zye01\Desktop\GBM\GBM_2\data\fiber_ratio_map_Gd_hypo_data','data');
x2 = data;
%h2 = histogram(x2,'Normalization','pdf','facecolor','blue');
h2 = histogram(x2,  'facecolor', 'blue');
h2.NumBins = 10;
%h1.BinWidth = 0.01;
hold on;
load('C:\Users\zye01\Desktop\GBM\GBM_2\data\fiber_ratio_map_FLAIR_Gd_data','data');
x3 = data;
%h3 = histogram(x3, 'Normalization', 'pdf', 'facecolor', [0 .5 0]);
h3 = histogram(x3, 'facecolor', [0 .5 0]);
h3.NumBins = 25;
%h1.BinWidth = 0.05;
hold on;
load('C:\Users\zye01\Desktop\GBM\GBM_2\data\fiber_ratio_map_Gd_hyper_data','data');
x1 = data;
%h1 = histogram(x1,'Normalization','pdf','facecolor','red');
h1 = histogram(x1, 'facecolor', 'red');
h1.NumBins = 25;
%h1.BinWidth = 0.05;
hold on;
% load('C:\Users\zye01\Desktop\GBM\data\fiber_ratio_map_ROI_NAWM_data','data');
% x4 = data;
% h4 = histogram(x4,'Normalization','pdf','facecolor','yellow');
% h4.NumBins = 20;
% %h1.BinWidth = 0.02
% hold on;

%plot curve fitting

[f,x2] = ksdensity(x2,'Bandwidth',0.01);
plot(x2,f,'LineWidth',3,'color','blue');
hold on;
[f,x3] = ksdensity(x3,'Bandwidth',0.04);
plot(x3,f,'LineWidth',3,'color','green');
hold on;
[f,x1] = ksdensity(x1,'Bandwidth',0.04);
plot(x1,f,'LineWidth',3,'color','red');
hold on;
% [f,x4] = ksdensity(x4,'Bandwidth',0.05);
% plot(x4,f,'LineWidth',3,'color','yellow');
% hold on;

axis([0 1 0 20]);
ax = gca;
ax.XTick = [0 0.5 1];
ax.YTick = [0 10 20];
ax.TickDir = 'out';
ax.TickDir = 'out';
%title('Anisotropic Fraction');
set(gca,'fontsize',20,'Fontweight','bold','linewidth',2);
box off;


%% DTI_ADC

% subplot(2,3,1);
% clear all;
% %plot histogram;
% load('C:\Users\zye01\Desktop\GBM\data\dti_adc_map_ROI_Gd_hyper_data','data');
% x3 = data;
% h3 = histogram(x3,'Normalization','pdf','facecolor','red');
% h3.NumBins = 40;
% %h1.BinWidth = 0.05
% hold on;
% 
% load('C:\Users\zye01\Desktop\GBM\data\dti_adc_map_ROI_Gd_hypo_data','data');
% x1 = data;
% h1 = histogram(x1,'Normalization','pdf','facecolor','blue');
% h1.NumBins = 40;
% %h1.BinWidth = 0.05
% hold on;
% load('C:\Users\zye01\Desktop\GBM\data\dti_adc_map_ROI_FLAIR_Gd_data','data');
% x2 = data;
% h2 = histogram(x2,'Normalization','pdf','facecolor',[0 .5 0]);
% h2.NumBins = 40;
% %h1.BinWidth = 0.05
% hold on;
% 
% 
% % load('C:\Users\zye01\Desktop\GBM\data\dti_adc_map_ROI_NAWM_data','data');
% % % x4 = data;
% % % h4 = histogram(x4,'Normalization','pdf','facecolor','yellow');
% % % h4.NumBins = 20;
% % % %h1.BinWidth = 0.02
% % hold on;
% legend('Gd-T1W Hyper','Gd-T1W Hypo','FLAIR Hyper');
% legend('boxoff');
% legend('Location','southwest');
% hold on;
% 
% %plot curve fitting
% [f,x1] = ksdensity(x1,'Bandwidth',0.04);
% plot(x1,f,'LineWidth',3,'color','blue');
% hold on;
% [f,x2] = ksdensity(x2,'Bandwidth',0.04);
% plot(x2,f,'LineWidth',3,'color','green');
% hold on;
% [f,x3] = ksdensity(x3,'Bandwidth',0.04);
% plot(x3,f,'LineWidth',3,'color','red');
% hold on;
% % [f,x4] = ksdensity(x4,'Bandwidth',0.05);
% % plot(x4,f,'LineWidth',3,'color','yellow');
% % hold on;
% 
% axis([0 3 0 2.5]);
% ax = gca;
% ax.XTick = [0 1 2 3];
% ax.YTick = [0 0.5 1 1.5 2 2.5];
% ax.TickDir = 'out';
% ax.TickDir = 'out';
% title('DTI ADC');
% set(gca,'fontsize',20,'Fontweight','bold','linewidth',2);
% box off;
% hold on;
% 
% %DTI FA
% clear all;
% subplot(2,3,2);
% %plot histogram;
% load('C:\Users\zye01\Desktop\GBM\data\dti_fa_map_ROI_Gd_hypo_data','data');
% x1 = data;
% h1 = histogram(x1,'Normalization','pdf','facecolor','blue');
% %h1.BinWidth = 0.005;
% h1.NumBins = 40;
% hold on;
% load('C:\Users\zye01\Desktop\GBM\data\dti_fa_map_ROI_FLAIR_Gd_data','data');
% x2 = data;
% h2 = histogram(x2,'Normalization','pdf','facecolor',[0 .5 0]);
% h2.NumBins = 80;
% %h1.BinWidth = 0.05;
% hold on;
% load('C:\Users\zye01\Desktop\GBM\data\dti_fa_map_ROI_Gd_hyper_data','data');
% x3 = data;
% h3 = histogram(x3,'Normalization','pdf','facecolor','red');
% h3.NumBins = 80;
% %h1.BinWidth = 0.03;
% hold on;
% % load('C:\Users\zye01\Desktop\GBM\data\dti_fa_map_ROI_NAWM_data','data');
% % x4 = data;
% % h4 = histogram(x4,'Normalization','pdf','facecolor','yellow');
% % h4.NumBins = 30;
% % %h1.BinWidth = 0.02
% % hold on;
% 
% %plot curve fitting
% [f,x1] = ksdensity(x1,'Bandwidth',0.005);
% plot(x1,f,'LineWidth',3,'color','blue');
% hold on;
% [f,x2] = ksdensity(x2,'Bandwidth',0.015);
% plot(x2,f,'LineWidth',3,'color','green');
% hold on;
% [f,x3] = ksdensity(x3,'Bandwidth',0.01);
% plot(x3,f,'LineWidth',3,'color','red');
% hold on;
% % [f,x4] = ksdensity(x4,'Bandwidth',0.05);
% % plot(x4,f,'LineWidth',3,'color','yellow');
% % hold on;
% 
% axis([0 0.5 0 20]);
% ax = gca;
% ax.XTick = [0 0.1 0.2 0.3 0.4 0.5];
% ax.YTick = [0 5 10 15 20];
% ax.TickDir = 'out';
% ax.TickDir = 'out';
% title('DTI FA');
% set(gca,'fontsize',20,'Fontweight','bold','linewidth',2);
% box off;
% hold on;
% 
% %Iso ADC
% clear all;
% subplot(2,3,3);
% %plot histogram;
% load('C:\Users\zye01\Desktop\GBM\data\iso_adc_map_ROI_Gd_hypo_data','data');
% x1 = data;
% h1 = histogram(x1,'Normalization','pdf','facecolor','blue');
% h1.NumBins = 40;
% %h1.BinWidth = 0.1;
% hold on;
% load('C:\Users\zye01\Desktop\GBM\data\iso_adc_map_ROI_FLAIR_Gd_data','data');
% x2 = data;
% h2 = histogram(x2,'Normalization','pdf','facecolor',[0 .5 0]);
% h2.NumBins = 40;
% %h1.BinWidth = 0.1;
% hold on;
% load('C:\Users\zye01\Desktop\GBM\data\iso_adc_map_ROI_Gd_hyper_data','data');
% x3 = data;
% h3 = histogram(x3,'Normalization','pdf','facecolor','red');
% h3.NumBins = 40;
% %h1.BinWidth = 0.1;
% hold on;
% % load('C:\Users\zye01\Desktop\GBM\data\iso_adc_map_ROI_NAWM_data','data');
% % x4 = data;
% % h4 = histogram(x4,'Normalization','pdf','facecolor','yellow');
% % h4.NumBins = 20;
% % %h1.BinWidth = 0.02
% % hold on;
% 
% %plot curve fitting
% [f,x1] = ksdensity(x1,'Bandwidth',0.04);
% plot(x1,f,'LineWidth',3,'color','blue');
% hold on;
% [f,x2] = ksdensity(x2,'Bandwidth',0.03);
% plot(x2,f,'LineWidth',3,'color','green');
% hold on;
% [f,x3] = ksdensity(x3,'Bandwidth',0.05);
% plot(x3,f,'LineWidth',3,'color','red');
% hold on;
% % [f,x4] = ksdensity(x4,'Bandwidth',0.05);
% % plot(x4,f,'LineWidth',3,'color','yellow');
% % hold on;
% 
% axis([0 3 0 2.5]);
% ax = gca;
% ax.XTick = [0 1 2 3];
% ax.YTick = [0 0.5 1 1.5 2 2.5];
% ax.TickDir = 'out';
% ax.TickDir = 'out';
% title('Iso ADC');
% set(gca,'fontsize',20,'Fontweight','bold','linewidth',2);
% box off;
% hold on;

%% plot histogram and curve fitting
% DTI_ADC on Gd-T1W hyper-intensity regions
% subplot(2,3,1);
% clear all;
% adc = {'dti_adc_map_ROI_FLAIR_Gd_data','dti_adc_map_ROI_Gd_hyper_data','dti_adc_map_ROI_Gd_hypo_data'};
% dirData = ['C:\Users\zye01\Desktop\GBM\data'];
% for i = 1:numel(adc);
%     adc_data = adc{i};
%     name = char(adc_data);
%     filename = fullfile(dirData,[name,'.mat']);
%     load(filename,'data');
%     %h1 = histogram(data,'Normalization','pdf','facecolor','red');
%     h = histogram(data,'Normalization','pdf');
%     %h1.BinEdges = [0;2];
%     h.NumBins = 20;
%     %h1.BinWidth = 0.02;
%     hold on;
%     x = data;
%     [f,x] = ksdensity(x,'Bandwidth',0.1);
%     %plot(x,f,'LineWidth',3,'color','red');
%     plot(x,f,'LineWidth',3);
%     axis([0 3 0 2.5]);
%     ax = gca;
%     ax.XTick = [0 1 2 3];
%     ax.YTick = [0 0.5 1 1.5 2 2.5];
%     ax.TickDir = 'out';
%     ax.TickDir = 'out';
%     title('DTI ADC');
%     set(gca,'fontsize',20,'Fontweight','bold','linewidth',2);
%     box off;
%     hold on;
% end;

%%

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
%         subplot(2,3,i)
%         h = histogram(data,'Normalization','pdf');
%         h3.NumBins = 20;
%         hold on;
%         [f,data] = ksdensity(data,'Bandwidth',0.05);
%         plot(data,f,'LineWidth',3);
%         hold on;
%         axis([0 3 0 2.5]);
%         ax = gca;
%         ax.XTick = [0 1 2 3];
%         ax.YTick = [0 0.5 1 1.5 2 2.5];
%         ax.TickDir = 'out';
%         ax.TickDir = 'out';
%         title(img);
%         set(gca,'fontsize',20,'Fontweight','bold','linewidth',2);
%         box off;
%         hold on;
%     end;
% end;
