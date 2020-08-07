A = xlsread('001','PCa');
B = xlsread('001','benign');

H_ADC = A(:,32);
F_ADC = A(:,40);
R_ADC = A(:,37);
HR_ADC = A(:,36);
DTI_ADC = A(:,3);

x_values = 0:0.001:4;


% PCa distribution
subplot(2,1,1);

pd = fitdist(DTI_ADC,'Normal');
DTI_ADC = pdf(pd,x_values);
plot(x_values,DTI_ADC,'LineWidth',5);
hold on;

pd = fitdist(HR_ADC,'Normal');
HR_ADC = pdf(pd,x_values);
plot(x_values,HR_ADC,'LineWidth',5);
hold on;

pd = fitdist(R_ADC,'Normal');
R_ADC = pdf(pd,x_values);
plot(x_values,R_ADC,'LineWidth',5);
hold on;

pd = fitdist(H_ADC,'Normal');
H_ADC = pdf(pd,x_values);
plot(x_values,H_ADC,'LineWidth',5);
hold on;

pd = fitdist(F_ADC,'Normal');
F_ADC = pdf(pd,x_values);
plot(x_values,F_ADC,'LineWidth',5);
hold on;

axis([0 4 0 10]);

%xlabel('PCa');
ylabel('Amplititude');
legend('DTI ADC','HR ADC','R ADC','H ADC','F ADC');
legend('boxoff');
legend('Location','southwest')

set(gca,'fontsize',20,'Fontweight','bold','linewidth',4);
hold off;
title('PCa');


% benign tissue distribution

subplot(2,1,2);

H_ADC = B(:,32);
F_ADC = B(:,40);
R_ADC = B(:,37);
HR_ADC = B(:,36);
DTI_ADC = B(:,3);

pd = fitdist(DTI_ADC,'Normal');
DTI_ADC = pdf(pd,x_values);
plot(x_values,DTI_ADC,'LineWidth',5);
hold on;

pd = fitdist(HR_ADC,'Normal');
HR_ADC = pdf(pd,x_values);
plot(x_values,HR_ADC,'LineWidth',5);
hold on;

pd = fitdist(R_ADC,'Normal');
R_ADC = pdf(pd,x_values);
plot(x_values,R_ADC,'LineWidth',5);
hold on;

pd = fitdist(H_ADC,'Normal');
H_ADC = pdf(pd,x_values);
plot(x_values,H_ADC,'LineWidth',5);
hold on;

pd = fitdist(F_ADC,'Normal');
F_ADC = pdf(pd,x_values);
plot(x_values,F_ADC,'LineWidth',5);
hold on;

axis([0 4 0 10]);

%xlabel('benign');
ylabel('Amplititude');

legend('DTI ADC','HR ADC','R ADC','H ADC','F ADC');
legend('boxoff');
legend('Location','southwest');

set(gca,'fontsize',20,'Fontweight','bold','linewidth',4);
hold off;
title('benign');

