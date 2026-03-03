function [s_i, v_j] = non_uniform_mesh(S_bar,K,v_bar,m_1,m_2)

% This function generates the non-uniform meshes for s and v as described
% in in't Hout and Foulon (2010). The transformation formulas are implemented
% exactly as presented in the reference.

    c = K/5;

    ind_i    = 0:m_1;
    delta_xi = (asinh((S_bar-K)/c) - asinh(-K/c)) / m_1;
    xi_i     = asinh(-K/c) + ind_i * delta_xi;
    s_i      = K + c * sinh(xi_i);   
    s_i      = s_i';

    d = v_bar/500;

    ind_j     = 0:m_2;
    delta_eta = asinh(v_bar/d) / m_2;
    eta_j     = ind_j * delta_eta;
    v_j       = d * sinh(eta_j);     
    v_j       = v_j';

end