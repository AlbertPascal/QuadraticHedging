function [V_t,phi_t,delta_phi_t] = heston_hedge(t,s,v,rho,sigma,FD_eval)

% This function computes the value process and quadratic optimal hedge ratio 
% for a given tuple (t,s,v). In addition, it returns the delta-hedging ratio.

    % function value and partial derivatives w.r.t. s and v (from FD evaluation)
    [u, u_s, u_v] = FD_eval(t,s,v);
    
    % value process
    V_t=u;
    
    % quadratic optimal hedge ratio
    phi_t=u_s+u_v.*(rho*sigma)./s;
    
    % delta-hedging ratio
    delta_phi_t=u_s;

end