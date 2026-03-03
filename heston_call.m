function[c]=heston_call(S,K,r,q,tau,v0,kappa,theta,sigma,rho)
    
% This function computes the Heston (1993) call option price using the
% semi-closed-form formula. The implementation follows the Little Heston
% Trap as described in Albrecher et al. (2007).

    x  = log(S);

    u  = [0.5; -0.5];
    a  = kappa*theta;
    b  = [kappa - rho*sigma; kappa];
    
    xi = @(phi) rho*sigma*1i .* phi;
    
    d  = @(phi,j) sqrt( (xi(phi) - b(j)).^2 ...
                        - sigma.^2 .* ( 2*u(j)*1i .* phi - phi.^2 ) );
    
    % Little Heston Trap: note the flipped signs in g, C, and D

    g  = @(phi,j) (b(j) - xi(phi) - d(phi,j)) ...
                 ./ (b(j) - xi(phi) + d(phi,j));
    
    C  = @(tau,phi,j) (r - q)*1i .* phi .* tau ...
        + (a./sigma.^2) .* ( (b(j) - xi(phi) - d(phi,j)) .* tau ...
        - 2 .* log( (1 - g(phi,j) .* exp(-d(phi,j).*tau)) ./ (1 - g(phi,j)) ) );
    
    D  = @(tau,phi,j) ((b(j) - xi(phi) - d(phi,j))./sigma.^2) ...
        .* ( (1 - exp(-d(phi,j).*tau)) ./ (1 - g(phi,j).*exp(-d(phi,j).*tau)) );
    
    cf = @(tau,phi,j) exp( C(tau,phi,j) + D(tau,phi,j).*v0 + 1i.*phi.*x );
    
    integrand_1 = @(phi) real( (exp(-1i.*phi.*log(K)) .* cf(tau,phi,1)) ./ (1i.*phi) );
    integrand_2 = @(phi) real( (exp(-1i.*phi.*log(K)) .* cf(tau,phi,2)) ./ (1i.*phi) );
    
    phi_max = 1000;  
    P1 = 0.5 + 1/pi * integral(integrand_1, 1e-8, phi_max); 
    P2 = 0.5 + 1/pi * integral(integrand_2, 1e-8, phi_max); 
    
    c = S*exp(-q*tau)*P1 - K*exp(-r*tau)*P2;
    
end