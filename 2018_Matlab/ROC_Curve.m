(* ::Package:: *)

% A = csvread('\10 .39.42.102\temp\2018_Legacy_Project','tumor_voxel.csv','sheet1');
A = csvread('tumor_voxel.csv');
pred_ 1 = A(:,1);
resp = [zeros(24420,1);ones(18178,1)];
mdl = fitglm(pred_ 1,resp,'Distribution','binomial','Link','logit');
scores_ 1 = mdl.Fitted.Probability;
%[AUC_ 1,OPTROCPT_ 1] = perfcurve(resp,scores_ 1,'1');
[X_ 1,Y_ 1,T_ 1,AUC_ 1,OPTROCPT_ 1] = perfcurve(resp,scores_ 1,'1');
sensitivity_ 1 = OPTROCPT_ 1(1,2);
specificity_ 1 = 1 - OPTROCPT_ 1(1,1);
plot(X_ 1,Y_ 1,'b','Linewidth',3);
set(gca,'linewidth',2,...
    'Fontsize',18,...
    'fontweight','bold');
legend('Restricted Fraction','D-Histo','FontSize',18,...
    'FontWeight','bold','Location','best');
xlabel('1-Specificity','FontSize',18,...
      'FontWeight','bold');
ylabel('Sensativity','FontSize',18,...
      'FontWeight','bold');
ax = gca;
ax.XTick = [0 0.2 0.4 0.6 0.8 1];
ax.YTick = [0 0.2 0.4 0.6 0.8 1];

% title('invasive ductal carcinoma vs. benign tissue','FontSize',16,...
      %'FontWeight','bold');
      % len_ 1 = length(A(:,1));
% types_ 1 = [repmat({'benign'},20,1);repmat({'cancer'},62,1)];
%[X_ 1,Y_ 1,T_ 1,AUC_ 1,OPTROCPT_ 1] = perfcurve(types_ 1,scores_ 1,'cancer');
% len_ 2 = length(pred_ 2);
% types_ 2 = [repmat({'benign'},20,1);repmat({'cancer'},62,1)]
%[X_ 2,Y_ 2,T_ 2,AUC_ 2,OPTROCPT_ 2] = perfcurve(types_ 2,scores_ 2,'cancer');


