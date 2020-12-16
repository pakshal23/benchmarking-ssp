function [x_est, opt_cost, num_iter] = l2_estimator(y, H, HTH, L, LTL, lambda, ~, ~)
%---------------------------------------------------------------------------------
% Implementation of the l2-regularized estimator
%
% Inputs: y: Data vector
%         H: Forward model
%         HTH: H'*H
%         L: Regularization matrix
%         LTL: L'*L
%         lambda: Regularization parameter
%
% Outputs: x_est: solution
%          opt_cost: optimal cost
%          num_iter: 0
%---------------------------------------------------------------------------------

HTy = H'*y;
x_est = (HTH + lambda*LTL) \ HTy;
opt_cost = norm(y - H*x_est)^2 + lambda*((norm(L*x_est))^2);
num_iter = 0;
end

