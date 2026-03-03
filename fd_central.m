function [beta_im1,beta_i0,beta_ip1]=fd_central(x)

% This function generates the beta coefficients for the central finite
% difference scheme.

    dx=diff(x);
    
    dx_p1=dx(2:end);
    dx=dx(1:end-1);
    
    % denominator for beta_{i,-1} and beta_{i,1}
    denom=dx+dx_p1;
    
    % beta coefficients
    beta_im1=-dx_p1./(dx.*denom);
    beta_i0=(dx_p1-dx)./(dx.*dx_p1);
    beta_ip1=dx./(dx_p1.*denom);

end