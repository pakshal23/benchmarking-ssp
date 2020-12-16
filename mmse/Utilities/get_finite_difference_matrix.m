function [D] = get_finite_difference_matrix(K)
%------------------------------------------------------------
% Inputs: K: dimension of the matrix / length of the signal
%
% Outputs: D:  [K x K] finite-difference matrix with zero-boundary conditions
%------------------------------------------------------------

D_inv = cumsum(eye(K));
D = inv(D_inv);

end

