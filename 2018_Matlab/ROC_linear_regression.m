A = xlsread('ROC','sheet2');
pred_1 = A(:,4);
pred_2 = A(:,1:8);
resp = [zeros(145,1);ones(261,1)];
mdl = fitglm(pred_1,resp,'Distribution','binomial','Link','logit');
scores_1 = mdl.Fitted.Probability;
%[AUC_1,OPTROCPT_1] = perfcurve(resp,scores_1,'1');
[X_1,Y_1,T_1,AUC_1,OPTROCPT_1] = perfcurve(resp,scores_1,'1');
mdl = fitglm(pred_2,resp,'Distribution','binomial','Link','logit');
scores_2 = mdl.Fitted.Probability;
%[AUC_2,OPTROCPT_2] = perfcurve(resp,scores_2,'1');
[X_2,Y_2,T_2,AUC_2,OPTROCPT_2] = perfcurve(resp,scores_2,'1');
sensitivity_1 = OPTROCPT_1(1,2);
specificity_1 = 1 - OPTROCPT_1(1,1);
sensitivity_2 = OPTROCPT_2(1,2);
specificity_2 = 1 - OPTROCPT_2(1,1);
plot(X_1,Y_1,'b','Linewidth',3);
hold on;
plot(X_2,Y_2,'r','LineWidth',3);
hold off;
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

%title('invasive ductal carcinoma vs. benign tissue','FontSize',16,...
      %'FontWeight','bold');
      %len_1 = length(A(:,1));
%types_1 = [repmat({'benign'},20,1);repmat({'cancer'},62,1)];
%[X_1,Y_1,T_1,AUC_1,OPTROCPT_1] = perfcurve(types_1,scores_1,'cancer');
%len_2 = length(pred_2);
%types_2 = [repmat({'benign'},20,1);repmat({'cancer'},62,1)]
%[X_2,Y_2,T_2,AUC_2,OPTROCPT_2] = perfcurve(types_2,scores_2,'cancer');


