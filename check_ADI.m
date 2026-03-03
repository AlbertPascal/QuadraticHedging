function ADI_logical = check_ADI(U,s_int,v_int,m1,m2,K,r,q,TT,kappa,theta,sigma,rho)

% This function compares the ADI solution with the semi-closed-form Heston
% (1993) prices on a selected moneyness range and returns a logical 
% accuracy flag.

    % s and v value grids
    [s_value,v_value]=ndgrid(s_int,v_int);
    
    % vectorized s and v variables (consistent with vech/stacking convention)
    s_var=kron(ones(m2,1),s_value(:,1));
    v_var=kron(v_value(1,:)',ones(m1,1));
    
    % index for moneyness range s/K in (0.5,1.5)
    idx=s_var>0.5*K & s_var<1.5*K & v_var>0 & v_var<1;
    
    % heston call option prices using semi-closed formula
    heston=@(s,v)heston_call(s, K, r, q, TT, v, kappa, theta, sigma, rho);
    
    % reference solution matrix
    U_check = arrayfun(heston, s_value, v_value); 
    
    % vectorized reference solution
    U_check = U_check(:);
    
    % result matrix: ADI solution vs. semi-closed form reference
    result=[U U_check];
     
    % restrict comparison to the specified moneyness range
    result=result(idx,:);
    
    % absolute difference in option prices
    difference=abs(diff(result')');
    
    % maximum pricing error
    max_diff=max(difference);

    % indicator for ADI accuracy with prescribed tolerance
   ADI_logical=max_diff<1e-2;

end
    

    
    