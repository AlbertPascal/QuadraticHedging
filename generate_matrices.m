function [A0,A1,A2]= generate_matrices(m1,m2,s,v,r,q,kappa,theta,sigma,rho)

% This function generates the matrices A0, A1, and A2 for required for the
% CS ADI scheme. See thesis for details.

    %generate stencils
    [D_s,D_ss,D_v,D_vv,D_sv]=generate_stencils(s,v);
    
    % interior s- and v-grids (for readability, the suffix '_int' is omitted)
    s=s(2:end);
    v=v(1:end-1);
    
    % idendity matrices
    Is = speye(m1);  
    Iv = speye(m2);
    
    % A0, A1, and A2 specification
    A0 = rho * sigma * kron(v,s) .* D_sv;
    A1 = 0.5 * kron(v,s.^2).*kron(Iv, D_ss) + ...
        (r-q) * kron(ones(m2,1),s) .* kron(Iv, D_s)-0.5*r*speye(m1*m2);
    A2 = 0.5 * sigma^2 *kron(v,ones(m1,1)).* kron(D_vv, Is) + ...
        kron(kappa*(theta-v), ones(m1,1)) .* kron(D_v, Is)-0.5*r*speye(m1*m2);

end