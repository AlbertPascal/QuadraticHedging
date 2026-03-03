function [delta_C_tilde,V_0,V_t] = hedging_err(paths,r,rho,sigma,FD_eval,hedging_strategy)

% This function computes the hedging error, i.e. the discounted cost
% process increments delta_C_tilde, for the selected hedging strategy
% ('quadratic' or 'delta'). It also returns the initial value V_0 and the
% value process V_t.

    % extract (t,S_t,v_t)
    t=paths(:,1,:);
    S_t=paths(:,2,:);
    v_t=paths(:,3,:);
    
    % money market account B_t
    B_t=exp(r.*t);
    
    % value process and hedge ratios (quadratic and delta)
    [V_t,phi_t,phi_delta_t]=heston_hedge(t,S_t,v_t,rho,sigma,FD_eval);

    % select hedge ratio according to chosen strategy
    if strcmp(hedging_strategy,'quadratic')
        phi=phi_t;
    elseif strcmp(hedging_strategy,'delta')
        phi=phi_delta_t;
    else
        error('no valid hedging strategy is chosen')
    end
    
    % money market account position
    eta_t=(V_t-phi.*S_t)./B_t;

     % preallocate locally self-financing value process V_bar_t (incl. initial value)
    V_bar_t=zeros(size(V_t));
    V_bar_t(1,:,:)=V_t(1,:,:);
    
    % locally self-financing portfolio evolution
    V_bar_t(2:end,:,:)=phi(1:end-1,:,:).*S_t(2:end,:,:)...
        +eta_t(1:end-1,:,:).*B_t(2:end,:,:);
    
    % discounted cost process increments
    delta_C_tilde=(V_t-V_bar_t)./B_t;

    % exclude increment at t=0 (delta_C_tilde = 0 by construction)
    delta_C_tilde=delta_C_tilde(2:end,:,:);

    % inital value
    V_0=V_t(1,:,:);
   

end