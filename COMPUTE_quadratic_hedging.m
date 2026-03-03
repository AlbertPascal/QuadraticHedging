%% COMPUTE_quadratic_hedging
% This script evaluates the hedging performance of three strategies:
%   (i)   a PDE-based quadratic optimal hedging strategy,
%   (ii)  the standard delta hedging strategy, and
%   (iii) a deep hedging strategy.
%
% For the PDE-based approach, the Alternating Direction Implicit (ADI)
% scheme proposed by in't Hout and Foulon (2010) is implemented to solve the 
% associated PDE and derive the corresponding hedge ratio.
%
% For the deep hedging approach, the script generates the required
% training and testing datasets. The neural network training and
% evaluation are performed separately using Python code.


% clear workspace
clear

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ADI Scheme
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% setup
K  = 100;
m1 = 200;
m2 = 200;
m  = m1*m2;
N  = 500;
TT = 1;
dt = TT/N;
S_bar  = 8*K;
v_bar  = 5;

% case specification
Case = menu('Choose case:', 'Case 1', 'Case 2');

switch Case
    case 1
        kappa=1.5;
        theta=0.04;
        sigma=0.3;
        rho=-0.9;
        r=0.025;
        q=0; 
    case 2
        kappa=3;
        theta=0.12;
        sigma=0.04;
        rho=0.6;
        r=0.01;
        q=0.0;
end

% check: Feller condition under Q
if 2*kappa*theta<sigma^2
    error('Feller condition under Q is not satisfied')
end

%% Grid, Stencils and Matrices

% grid
[s,v]=non_uniform_mesh(S_bar,K,v_bar,m1,m2);

% interior meshes
[s_int,v_int]=deal(s(2:end),v(1:end-1));

% stencils
[D_s,D_ss,D_v,D_vv,D_sv]=generate_stencils(s,v);

% matrices
[A0,A1,A2]=generate_matrices(m1,m2,s,v,r,q,kappa,theta,sigma,rho);

%% Boundaries and Initial Condition

% Delta_s for ghost point
Delta_s=s(end)-s(end-1);

% beta and delta coefficients at the upper boundary, i.e. at m_2-1,1
dv = diff(v(end-2:end));
[dv_m1, dv_m] = deal(dv(1), dv(2));
beta=dv_m1./(dv_m.*(dv_m1+dv_m));
delta=2./(dv_m.*(dv_m1+dv_m));

% build_boundaries function 
build_boundaries=@(tau)boundaries(tau,m1,m2,s_int,v_int,S_bar,r,q,beta,delta,sigma,rho,Delta_s);

% initial condition
payoff=max(s_int-K,0);
U0=repmat(payoff,m2,1);

%% CS Scheme and Derivatives

% CS scheme
[U,U_grid]=CS(m,A0,A1,A2,U0,N,dt,build_boundaries);

% U_arrays - values and derivatives
[U_array,U_s_array,U_v_array]=U_arrays(m1,m2,U_grid,D_s,D_v,q,N,dt,TT);

% t vector (forward in time)
t_vec=0:dt:TT;

% linear interpolation of finite differences (FD) generated U_arrays
FD_u  = griddedInterpolant({s_int, v_int, t_vec}, U_array,   'linear', 'linear');
FD_us = griddedInterpolant({s_int, v_int, t_vec}, U_s_array, 'linear', 'linear');
FD_uv = griddedInterpolant({s_int, v_int, t_vec}, U_v_array, 'linear', 'linear');

% Finite differences (FD) evaluation function providing u, u_s, and u_v
FD_eval = @(t,s,v) deal(FD_u(s,v,t),FD_us(s,v,t),FD_uv(s,v,t));

%% Sanity Check

% ADI sanity check
ADI_logical=check_ADI(U,s_int,v_int,m1,m2,K,r,q,TT,kappa,theta,sigma,rho);

if ~ADI_logical
    error(['Maximal difference between option prices of the ADI scheme' ...
       ' and the semi-closed formula is not of order 10^-3 ']);
end

%% Plots

set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex'); 

[s_grid,v_grid]=meshgrid(s_int,v_int);

fig=figure('Position', [100 100 500 1500]);

tlo = tiledlayout(fig,2,1,'TileSpacing','compact','Padding','compact');

nexttile
t = 1;
surf(s_grid, v_grid, U_array(:,:,t).');
xlabel('$s$'); 
ylabel('$v$','Rotation',0);
zlabel('$u$','Rotation',0);
title('Option price at $t=0$ : $u(0,s,v)$');
xlim([0 2*K]);
ylim([0,1])
zlim([0 1.5*K]);
clim([0 1*K]);

nexttile
t = N+1;
surf(s_grid, v_grid, U_array(:,:,t).');
xlabel('$s$'); 
ylabel('$v$','Rotation',0);
zlabel('$u$','Rotation',0);
title('Option payoff at maturity $t=T$ : $u(T,s,v)$');
xlim([0 2*K]);
ylim([0 1]);
zlim([0 1.5*K]);
clim([0 1*K]);

filename=strcat('ADI_illustration_','Case_',string(Case),'.eps');

set(fig,'PaperPositionMode','auto');
exportgraphics(fig,filename,'ContentType','vector')

close(fig)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Hedging Simulation and Analysis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% setup
S0 = 100;
v0 = 0.05;
mu = 0.05;

% Q specification
Q=menu('Specify Q:','Q_star','Q');
switch Q
    case 1
        lambda=0;
        Q_measure='Q_star';
    case 2
        lambda=1;
        Q_measure='Q';
end

% trading time grid specification
N_sim=[50,100,250];
dt_sim=TT./N_sim;

% obtain P-parameters (kappa and theta)
[kappa_P,theta_P]=coefficients_Q2P(mu,r,kappa,theta,sigma,rho,lambda);

% check: Feller condition under P
if 2*kappa_P*theta_P<sigma^2
    error('Feller condition under P is not satisfied')
end

% performance measures function
performance_measures =@(x,y) [sum(x,1), abs(sum(x,1))./y, sum(abs(x),1), max(abs(x),[],1)];

% preallocated perfmance measures matrix
performance=zeros(3,8,3);

% number of Monte Carlo simulations
M=1e5;

%% NN training data

rng(1)

for i=1:numel(N_sim)

    dt_i=dt_sim(i);

    % heston paths for training
    paths_NN_training=heston_paths(S0,TT,dt_i,mu,v0,kappa_P,theta_P,sigma,rho,M);

    % t,B_t,S_t,v_t values required for computing V_t and discounting
    t_NN=paths_NN_training(:,1,:);
    B_t_NN=exp(r.*t_NN);
    S_t_NN=paths_NN_training(:,2,:);
    v_t_NN=paths_NN_training(:,3,:);

    % V_t values for training
    [V_t_NN_training,~,~]=heston_hedge(t_NN,S_t_NN,v_t_NN,rho,sigma,FD_eval);

    % S_t and V_t discounting
    paths_NN_training(:,2,:)=paths_NN_training(:,2,:)./B_t_NN;
    V_t_NN_training=V_t_NN_training./B_t_NN;

    filename=strcat('data_NN_training_','Case_',string(Case),'_',Q_measure,'_N_',string(N_sim(i)),'.mat');
    save(filename,'paths_NN_training','V_t_NN_training')

end


%% Hedging Analysis

rng(123);

% delta and quadratic hedging
for i=1:numel(N_sim)

    dt_i=dt_sim(i);

    % heston paths
    paths=heston_paths(S0,TT,dt_i,mu,v0,kappa_P,theta_P,sigma,rho,M);

    % discounted cost process and hedging performance from quadratic hedging
    [delta_C_tilde,V_0,V_t]=hedging_err(paths,r,rho,sigma,FD_eval,'quadratic');
    performance_quadratic=performance_measures(delta_C_tilde,V_0);
    
    % discounted cost process and hedging performance from delta hedging
    [delta_C_tilde,~,~]=hedging_err(paths,r,rho,sigma,FD_eval,'delta');
    performance_delta=performance_measures(delta_C_tilde,V_0);
    
    % deep hedging:t,B_t,S_t,v_t values required for computing V_t and (discounted) dC_t 
    t=paths(:,1,:);
    B_t=exp(r.*t);
    paths_NN_testing=paths;
    paths_NN_testing(:,2,:)=paths_NN_testing(:,2,:)./B_t;
    V_t_NN_testing=V_t./B_t;

    % save testing data, hedging via NN is performed in Python
    filename=strcat('data_NN_testing_','Case_',string(Case),'_',Q_measure,'_N_',string(N_sim(i)),'.mat');
    save(filename,'paths_NN_testing','V_t_NN_testing')

    % aggregate hedging performance measures for quadratic and delta hedging
    performance_quadratic_mean= mean(performance_quadratic,3);
    performance_quadratic_std = std(performance_quadratic,[],3);
    performance_delta_mean    = mean(performance_delta,3);
    performance_delta_std     = std(performance_delta,[],3);
    

    % hedging performance measures matrix for delta and quadratic hedging
    performance(1:2,:,i)=[performance_quadratic_mean, performance_quadratic_std;...
        performance_delta_mean, performance_delta_std];

end

% MATLAB pauses here. Run the Python deep-hedging code now (it will write dC_pred_*.mat),
% then click "Yes" to continue.
if menu('Did the Python code run?', 'Yes', 'No') ~= 1
    error('Python code must run first.');
end

% verify Python output files exist
for i = 1:numel(N_sim)
    fname = strcat('dC_pred_','Case_',string(Case),'_',Q_measure,'_N_',string(N_sim(i)),'.mat');
    if ~isfile(fname)
        error('Missing Python output file: %s', fname);
    end
end

% deep hedging
for i=1:numel(N_sim)

     % load predicted dC_t (in discounted units)
    filename=strcat('dC_pred_','Case_',string(Case),'_',Q_measure,'_N_',string(N_sim(i)),'.mat');
    load(filename);

    % discounted cost process and hedging performance from deep hedging
    delta_C_tilde=dC_NN;
    performance_deep=performance_measures(delta_C_tilde,V_0);

    % aggregate hedging performance measures for deep hedging
    performance_deep_mean = mean(performance_deep,3);
    performance_deep_std  = std(performance_deep,[],3);

    % hedging performance measures matrix (deep)
    performance(3,:,i)=[performance_deep_mean,performance_deep_std];

end

% save final hedging performance matrix
filename=strcat('hedging_performance_Case_',string(Case),'_',Q_measure);
save(filename,"performance")

% clear workspace
clear







