%% Script to compare FISTA vs ADMM

rng(0);

K = 100;
experiment = 'deconv_gaussian';  % 'deconv_gaussian' or 'deconv_airy_disk' or 'fourier_samp'
exp_param = 4;     % parameter for the chosen experiment (see generate_datasets.m for more details)
noise_level = 30;   % input SNR
[H, M] = get_measurement_matrix(experiment, exp_param, K);

handles.Prior = 'bernoulli-laplace';      % 'student' or 'alpha-stable' or 'bernoulli-laplace'
handles.K = K;
handles.Dist_Param = [0.6, 1];

D = get_finite_difference_matrix(handles.K);

sig = generate_discrete_levy_process(handles);
pow_Hx = mean((H*sig).^2);
noise_var = pow_Hx/(10^(noise_level/10));  

meas = H*sig + sqrt(noise_var)*randn([M, 1]);

% Parameters for the l1 and log estimators (iterative schemes)
fista_params.name = 'fista';      % 'fista' or 'admm'
fista_params.maxiter = 1e5;
fista_params.relative_tol = 1e-10;
fista_params.verbose = 1;
fista_params.ItUpOut = 10;
fista_params.iterVerb = 10;

admm_params.name = 'admm';      % 'fista' or 'admm'
admm_params.maxiter = 1e5;
admm_params.relative_tol = 1e-10;
admm_params.verbose = 1;
admm_params.ItUpOut = 10;
admm_params.iterVerb = 10;

x0 = zeros([handles.K,1]);  % initialization for the iterative schemes
lambda_list = [10.^(-5)];
l1_estimator_fun = @(y_val, lambda_val, params, x0_val) l1_estimator(y_val, H, D, lambda_val, params, x0_val);

[x_l1_fista, err_l1_fista, opt_cost_l1_fista, num_iter_l1_fista] = tune_lambda(l1_estimator_fun, meas, sig, lambda_list, [], fista_params, x0);
[x_l1_admm, err_l1_admm, opt_cost_l1_admm, num_iter_l1_admm] = tune_lambda(l1_estimator_fun, meas, sig, lambda_list, [], admm_params, x0);
