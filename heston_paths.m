function paths = heston_paths(S0,TT,dt,mu,v0,kappa,theta,sigma,rho,M)

% This function generates M paths following the dynamics of the Heston
% (1993) model. For the asset price S, a log-transformation is applied to
% use the Euler scheme without loss of precision relative to the Milstein
% scheme. For the variance process v, the Milstein scheme is used.
% See thesis for further details.

    t=0:dt:TT;

    N = round(TT/dt);

    Z1 = randn(N, M);
    Z2 = randn(N, M);
    Zv = rho .* Z1 + sqrt(1 - rho^2) .* Z2; 

    S = zeros(N+1, M);
    Y = zeros(N+1, M);
    v = zeros(N+1, M);

    S(1, :) = S0;
    Y(1, :) = log(S0);
    v(1, :) = v0;

    for i = 1:N
     
        v_i = v(i,:);

        Y(i+1,:) = Y(i,:)+(mu-0.5.*v_i)*dt+sqrt(v_i*dt).*Z1(i,:);

        S(i+1, :) = exp(Y(i+1,:));

        v_i1 = v_i+ kappa.*(theta-v_i)*dt+sigma.*sqrt(v_i*dt).*Zv(i, :)...
            +0.25*sigma^2*dt.*(Zv(i,:).^2-1);

        v(i+1, :) = max(v_i1, 0);
    end

    paths = zeros(N+1, 3, M);
    paths(:, 1, :) = repmat(t, [1, 1, M]);
    paths(:, 2, :) = S;
    paths(:, 3, :) = v;
    
end