function [all_x, all_err, all_cost, all_num_iter] = tune_lambda(estimator, y, gt, lambda_list, rho_list, params, x0)
%------------------------------------------------------------------------------------------
% This routine takes an estimator and runs it for the provided values of
% lambda (regularization parameter)
%
% Inputs: estimator: function handle for the estimator that you want to run 
%         y: measurement vector
%         gt: ground-truth signal
%         lambda_list: values of the regularization parameter
%         rho_list: values of rho for admm
%         params: optimization parameters (to be passed to the estimator)
%         x0: initialization
%
% Output: all_x: reconstructions for all the values of lambda
%         all_err: error for all the values of lambda
%         all_cost: optimal cost values for all the values of lambda
%         all_num_iter: number of iterations for convergence for all the values of lambda
%-------------------------------------------------------------------------------------------

num_lambda = length(lambda_list);

all_x = cell([num_lambda, 1]);
all_err = zeros([num_lambda, 1]);
all_cost = zeros([num_lambda, 1]);
all_num_iter = zeros([num_lambda, 1]);

if (isempty(rho_list))
    rho_list = lambda_list;
end
   
for i = 1:num_lambda
    params.rho = rho_list(i);  % For FISTA, we currently rely on the library to compute gam on its own
    [all_x{i,1}, all_cost(i,1), all_num_iter(i,1)] = estimator(y, lambda_list(i), params, x0);
    all_err(i,1) = norm(all_x{i,1} - gt)^2;
end
    
end
