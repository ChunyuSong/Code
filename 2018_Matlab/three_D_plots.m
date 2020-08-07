figure;
A = xlsread('ROC','PCa');
RF1 = A(:,4);
HF1 = A(:,5);
FF1 = A(:,6);

B = xlsread('ROC','benign_NB_PZ');
C = xlsread('ROC','benign_B_PZ'); 
RF_2 = B(:,4);
RF_3 = C(:,4);
RF2 = [RF_2;RF_3];

HF_2 = B(:,5);
HF_3 = C(:,5);
HF2 = [HF_2;HF_3];

FF_2 = B(:,6);
FF_3 = C(:,6);
FF2 = [FF_2;FF_3];

D = xlsread('ROC','benign_NB_TZ');
E = xlsread('ROC','benign_B_TZ');
RF_4 = D(:,4);
RF_5 = E(:,4);
RF3 = [RF_4;RF_5];

HF_4 = D(:,5);
HF_5 = E(:,5);
HF3 = [HF_4;HF_5];

FF_4 = D(:,6);
FF_5 = E(:,6);
FF3 = [FF_4;FF_5];

scatter3(HF1, FF1, RF1, 'MarkerEdgeColor',[0 0 0],...
    'MarkerFaceColor',[1 0 0],...
    'LineWidth',1.5);
hold on;

scatter3(HF2, FF2, RF2, 'MarkerEdgeColor',[0 0 0],...
    'MarkerFaceColor',[0 0 1],...
    'LineWidth',1.5);
hold on;

scatter3(HF3, FF3, RF3, 'MarkerEdgeColor',[0 0 0],...
    'MarkerFaceColor',[0 1 0],...
    'LineWidth',1.5);
hold on;

axis([0 1 0 1 0 0.5]);
zlabel('Restricted Fraction');
ylabel('Hindered Fraction');
xlabel('Fiber Fraction');
legend('PCa','BPZ','BTZ');
legend('Location','northeast');
set(gca,'fontsize',20,'Fontweight','bold','linewidth',4);
title('PCa vs. BPZ vs. BTZ');
hold off;





