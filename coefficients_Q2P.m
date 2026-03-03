function [kappa_P,theta_P] = coefficients_Q2P(mu,r,kappa_Q,theta_Q,sigma,rho,lambda)

% This function returns the parameters kappa and theta under the physical
% measure P.

    % specification of kappa and theta under P (see thesis for details)
    kappa_P=kappa_Q+sqrt(1-rho^2)*sigma*lambda;
    theta_P=(kappa_Q*theta_Q+rho*sigma*(mu-r))./kappa_P;

end