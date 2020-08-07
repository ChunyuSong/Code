figure;
A = xlsread('sum','BPZ');
x1 = A(:,5);
y1 = A(:,8);
z1 = A(:,4);

B = xlsread('sum','S_BPH');
x2 = B(:,5);
y2 = B(:,8);
z2 = B(:,4);

C = xlsread('sum','BPH');
x3 = C(:,5);
y3 = C(:,8);
z3 = C(:,4);

D = xlsread('sum','PCa');
x4 = D(:,5);
y4 = D(:,8);
z4 = D(:,4);

scatter3(x1, y1, z1, 'MarkerEdgeColor',[0 0 0],...
    'MarkerFaceColor',[1 0 0],...
    'LineWidth',1.5);
hold on;

scatter3(x2, y2, z2, 'MarkerEdgeColor',[0 0 0],...
    'MarkerFaceColor',[0 0 1],...
    'LineWidth',1.5);
hold on;

scatter3(x3, y3, z3, 'MarkerEdgeColor',[0 0 0],...
    'MarkerFaceColor',[0 1 0],...
    'LineWidth',1.5);
hold on;

scatter3(x4, y4, z4, 'MarkerEdgeColor',[0 0 0],...
    'MarkerFaceColor',[0 0.5 0],...
    'LineWidth',1.5);
hold on;

axis([0 0.8 0 0.8 0 0.5]);
xlabel('Hindered Fraction');
ylabel('Fiber Fraction');
zlabel('Restricted Fraction');

legend('BPZ','Stromal BPH','BPH','PCa');
legend('Location','northeast');
set(gca,'fontsize',20,'Fontweight','bold','linewidth',4);
title('BPZ vs. SBPH vs. BPH vs. PCa');
hold off;





