%% COMPUTE_LaTeX_tables
% This script generates (or reads from output files) the numerical values
% used in the thesis LaTeX tables and exports them in LaTeX-ready format.
% The table structure itself is specified in the thesis LaTeX source.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% LOAD HEADING PERFORMANCE MATRICES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Case specification
Case = menu('Choose case:', 'Case 1', 'Case 2');

switch Case
    case 1
        Case='Case_1';
    case 2
        Case='Case_2';
end

% Q specification
Q=menu('Specify Q:','Q_star','Q');
switch Q
    case 1
        Q_measure='Q_star';
    case 2
        Q_measure='Q';
end

% load data
filename=['hedging_performance_',Case,'_',Q_measure,'.mat'];
load(filename);

% performance matrix
A=performance;

% number of trading time points
N = [50 100 250];

% hedging strategies/methods
methods = {'Quadratic','Delta','Deep'};

for k = 1:size(A,3)

    % header: N specification
    fprintf('\\multicolumn{10}{l}{$N = %d$} \\\\ \n', N(k));

    % rows: methods and corresponding values
    for i = 1:size(performance,1)
        fprintf('& %s & ', methods{i});           % method name
         fprintf('%.3f & ', A(i,1:end-1,k));      % formatted values
        fprintf('%.3f \\\\\n', A(i,end,k));       % last value      
    end
 
end














