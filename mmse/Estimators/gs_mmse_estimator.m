function [x_est] = gs_mmse_estimator(y, H, L, u0, sig_params, algo_params)
%--------------------------------------------------------------------------------------
% Implementation of Gibbs sampling based MMSE estimators for student, alpha-stable 
% and bernoulli-laplace signals
%
% Inputs: y: Data vector
%         H: Forward model
%         L: Regularization matrix
%         u0: initialization
%         sig_params:  - type: 'student' or 'alpha-stable' or 'bernoulli-laplace'
%                      - dist_param: parameters describing the distribution of the signal               
%                                    (see generate_discrete_levy_process.m for details)
%                      - noise_var: variance of the noise
%         algo_params: - num_iter
%                      - burn_in
%
% Outputs: x_est: MMSE estimate
%--------------------------------------------------------------------------------------

if (strcmp(sig_params.type, 'student'))
    x_est = mmse_student(y, H, L, u0, sig_params, algo_params);
elseif (strcmp(sig_params.type, 'alpha-stable'))
    x_est = mmse_stable(y, H, L, u0, sig_params, algo_params);
elseif (strcmp(sig_params.type, 'bernoulli-laplace'))
    x_est = mmse_bernoulli_laplace(y, H, L, u0, sig_params, algo_params);
end

end