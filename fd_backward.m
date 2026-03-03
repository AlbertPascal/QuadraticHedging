function [alpha_im2, alpha_im1,alpha_i0]=fd_backward(x)

% This function generates the alpha coefficients for the backward finite
% difference scheme.

    x=x(1:end-1);
    dx=diff(x);
    
    dx_m1=dx(1:end-1);
    dx=dx(2:end);
    
    % denominator for alpha_{i,-2} and alpha_{i,0}
    denom=dx_m1+dx;
    
    % alpha coefficients
    alpha_im2=dx./(dx_m1.*denom);
    alpha_im1=(-dx_m1-dx)./(dx_m1.*dx);
    alpha_i0=(dx_m1+2.*dx)./(dx.*denom);

end