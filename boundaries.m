function [b0,b1,b2]=boundaries(t,m1,m2,s_int,v_int,S_bar,r,q,beta,delta,sigma,rho,Delta_s)

% This function constructs the boundary vectors for the PDE, as specified 
% in the thesis.

    % value v_{m2-1}, i.e. v_{-1} (denoted as v_m1)
    v_m1=v_int(end);

    % b0: boundary vector for mixed derivatives
    b0=zeros(m1*m2,1);
    b0(end-m1+1:end)=s_int*rho*sigma*v_m1*beta*exp(-q*t);

    % b11: boundary vector for the first derivative wrt s
    b11=zeros(m1,1);
    b11(end)=(r-q)*S_bar*exp(-q*t);
    b11=kron(ones(m2,1),b11);

    % b12: boundary vector for the second derivative wrt s
    b12=zeros(m1,1);
    b12(end)=0.5*S_bar^2*exp(-q*t)./Delta_s;
    b12=kron(v_int,b12);

    % b1: boundary vector for derivatives wrt s
    b1=b11+b12;

    % b2: boundary vector for derivatives wrt v
    b2=zeros(m1*m2,1);
    b2(end-m1+1:end)=(0.5*sigma^2*v_m1*delta)*exp(-q*t).*s_int;

end
