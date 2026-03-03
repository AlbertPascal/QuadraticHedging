function [D_s,D_ss,D_v,D_vv,D_sv]=generate_stencils(s,v)

% This function generates the finite-difference stencil matrices required
% to construct A0, A1, and A2 for the CS ADI scheme.
% See thesis for details.

    % beta and delta coefficients for central schemes (first and second derivative)
    [b_m1,b_0,b_p1]=fd_central(s);
    [d_m1,d_0,d_p1]=fd_central_second(s);
    
    % Delta_s required for ghost point
    Delta_s=s(end)-s(end-1);
    
    % D_s and D_ss stencils
    D_s=diag([b_m1(2:end);0],-1)+diag([b_0;0])+diag(b_p1,+1);
    D_ss=diag([d_m1(2:end);1./Delta_s^2],-1)+diag([d_0;-1./Delta_s^2])+diag(d_p1,+1);
    
    % dv values for the gamma coefficients of the forward scheme (first row of D_v )
    dv = diff(v(1:3));
    [dv1, dv2] = deal(dv(1), dv(2));
    
    % gamma coefficient for forward scheme
    g_0=(-2*dv1-dv2)./(dv1*(dv1+dv2));
    g_1=(dv1+dv2)./(dv1.*dv2);
    g_2=-dv1./(dv2*(dv1+dv2));
    
    % coefficients for backward, central, and central_second schemes (v-derivatives)
    [a_m2,a_m1,a_0]=fd_backward(v);
    [b_m1,b_0,b_p1]=fd_central(v);
    [d_m1,d_0,d_p1]=fd_central_second(v);
    
    % D_v stencil based on backward scheme
    D_v_backward=diag(a_m2,-2)+diag([0;a_m1],-1)+diag([0;0;a_0]);
    
    % D_v stencil based on central scheme (with forward scheme in first row)
    D_v=diag(b_m1,-1)+diag([g_0;b_0])+diag([g_1;b_p1(1:end-1)],+1);
    D_v(1,3)=g_2;
    
    % D_v modified stencil for mixed-derivative terms (used for constructing D_sv)
    D_v_mod=D_v;
    D_v_mod(1,1:3)=zeros(1,3);
    
    % index indicating v > 1 (switch to backward scheme)
    idx=v(1:end-1)>1;
    
    % D_v stencil; apply backward scheme for v > 1
    D_v(idx,:)=D_v_backward(idx,:);
    
    % D_vv stencil
    D_vv=diag(d_m1,-1)+diag([0;d_0])+diag([0;d_p1(1:end-1)],+1);
    
    % D_sv stencil
    D_sv=kron(D_v_mod,D_s);
    
    % stencils as sparse matrices 
    D_s  = sparse(D_s);
    D_ss = sparse(D_ss);
    D_v  = sparse(D_v);
    D_vv = sparse(D_vv);
    D_sv = sparse(D_sv);

end