function [U_array,U_s_array,U_v_array]=U_arrays(m1,m2,U_grid,D_s,D_v,q,N,dt,TT)

% This function constructs arrays of U and its partial derivatives w.r.t.
% s and v by applying the same stencil matrices D_s and D_v used in the
% construction of the ADI matrices A0, A1, and A2.

    % tau mesh (stored backward in time)
    tau_mesh=TT:-dt:0;

    % boundary contribution for U_s
    b_S=sparse(m1,N+1);
    b_S(end,:)=exp(-q*tau_mesh);
    b_S_grid=kron(ones(m2,1),b_S);
    
    % s-derivative grid
    U_s_grid=kron(speye(m2),D_s)*U_grid+b_S_grid;

    % v-derivative grid
    U_v_grid=kron(D_v,speye(m1))*U_grid;
    
   % preallocate arrays
    U_array = zeros(m1,m2,N+1);
    U_s_array = zeros(m1,m2,N+1);
    U_v_array = zeros(m1,m2,N+1);
   
    % reshape column-stacked grids to 3D arrays
    for t = 1:N+1
        U_array(:,:,t) = reshape(U_grid(:,t), m1, m2);
        U_s_array(:,:,t) = reshape(U_s_grid(:,t), m1, m2);
        U_v_array(:,:,t) = reshape(U_v_grid(:,t), m1, m2);
    end

end