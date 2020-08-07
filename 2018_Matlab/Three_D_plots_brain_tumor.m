figure;
ax = gca;

A = xlsread('all_true', '1');
x1 = A(:, 3);
y1 = A(:, 4);
z1 = A(:, 5);

B = xlsread('all_true', '2');
x2 = B(:, 3);
y2 = B(:, 4);
z2 = B(:, 5);

C = xlsread('all_true', '3');
x3 = C(:, 3);
y3 = C(:, 4);
z3 = C(:, 5);

scatter3(x1, y1, z1, 42, 'MarkerEdgeColor', [0 0 0],...
    'MarkerFaceColor', [1 0 0],...
    'LineWidth', 1);
hold on;

scatter3(x2, y2, z2, 42, 'MarkerEdgeColor', [0 0 0],...
    'MarkerFaceColor', [0 0 1],...
    'LineWidth', 1);
hold on;

scatter3(x3, y3, z3, 42, 'MarkerEdgeColor', [0 0 0],...
    'MarkerFaceColor', [0 1 0],...
    'LineWidth', 1);
hold on;

ax.FontSize = 16;
ax.TickDir = 'out';

axis([0 1 0 1 0 0.5]);
ax.XTick = [0 0.2 0.4 0.6 0.8 1];
ax.YTick = [0 0.2 0.4 0.6 0.8 1];
ax.ZTick = [0 0.1 0.2 0.3 0.4 0.5];

set(gca, 'linewidth', 3, 'fontweight', 'bold', 'fontsize', 16);

% xlabel('Restricted Fraction');
% ylabel('Hindered Fraction');
% zlabel('Anisotropic Fraction');

daspect([1 1 0.5]);

% legend('BPZ','Stromal BPH','BPH','PCa');
% legend('Location','northeast');
% set(gca,'fontsize',20,'Fontweight','bold','linewidth',4);
% title('BPZ vs. SBPH vs. BPH vs. PCa');
hold off;





