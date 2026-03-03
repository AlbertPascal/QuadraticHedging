%% COMPUTE_non_uniform_grid_plot
% This script generates the non-uniform grid plot presented in in't Hout
% and Foulon (2010).

% LaTeX setup
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex'); 

% general setup
K=100;
m_1=30;
m_2=15;
S_bar=8*K;
v_bar=5;

% non-uniform mesh and grid
[s,v]=non_uniform_mesh(S_bar,K,v_bar,m_1,m_2);
[sM,vM]=meshgrid(s,v);

% figure
figure;
plot(sM,vM,'k');
hold on;
plot(sM',vM','k');
title('Non-uniform grid ($\\m_1=30, \\m_2=15, K=100$)')
xlabel('$s$');
ylabel('$v$','Rotation',0)
xlim([s(1),s(end)])

exportgraphics(gca,'non-uniform_grid.eps')