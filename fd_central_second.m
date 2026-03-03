function [delta_im1,delta_i0,delta_ip1]=fd_central_second(x)

% This function generates the delta coefficients for the central finite
% difference scheme approximating the second derivative.

    dx=diff(x);
    
    dx_p1=dx(2:end);
    dx=dx(1:end-1);
    
    % denominator for delta_{i,-1} and delta_{i,1}
    denom=dx+dx_p1;
    
    % delta coefficients
    delta_im1=2./(dx.*denom);
    delta_i0=-2./(dx.*dx_p1);
    delta_ip1=2./(dx_p1.*(denom));

end