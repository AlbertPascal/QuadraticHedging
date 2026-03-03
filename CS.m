function [U,U_grid] =CS(m,A0,A1,A2,U0,N,dtau,build_boundaries)

% This function performs the Craig–Sneyd (CS) ADI scheme and returns the 
% solution at t = 0 as well as the full space–time solution grid. 
% Details of the scheme are provided in the thesis.

    % theta-method parameter (theta = 1/2 is the optimal choice)
    theta = 0.5;

    % system matrices for implicit steps
    M1 = speye(m) - theta*dtau*A1;      
    M2 = speye(m) - theta*dtau*A2; 
    
    % LU decompositions
    [L1,U1,P1,Q1] = lu(M1); 
    [L2,U2,P2,Q2] = lu(M2);
    
    % auxiliary solvers based on LU decompositions
    solve1 = @(rhs) Q1*(U1\(L1\(P1*rhs)));
    solve2 = @(rhs) Q2*(U2\(L2\(P2*rhs)));

    % preallocate space–time solution grid
    U_grid=zeros(m,N+1);
    U_grid(:,end)=U0;

    % initialize solution
    U=U0;

    for n = 1:N

        % current and previous time levels
        tau_nm1 = (n-1)*dtau;  
        tau_n = n*dtau;

        % boundary contributions
        [b0m,b1m,b2m] = build_boundaries(tau_nm1);
        [b0p,b1p,b2p] = build_boundaries(tau_n);

        % Craig–Sneyd ADI scheme (see thesis for details)
        Fm = (A0 + A1 + A2) * U + b0m + b1m + b2m;    
        Y0 = U + dtau*Fm;
    
        rhs1 = Y0 + theta*dtau*(-A1*U + b1p - b1m);
        Y1   = solve1(rhs1);
    
        rhs2 = Y1 + theta*dtau*(-A2*U + b2p - b2m);
        Y2   = solve2(rhs2);
    
        Yt0  = Y0 + 0.5*dtau*(A0*Y2 + b0p - A0*U - b0m);       
    
        rhs1 = Yt0 + theta*dtau*(-A1*U + b1p - b1m);
        Yt1  = solve1(rhs1);
    
        rhs2 = Yt1 + theta*dtau*(-A2*U + b2p - b2m);
        U    = solve2(rhs2);

        % store solution at current time level
        U_grid(:,N+1-n)=U;

    end


end
