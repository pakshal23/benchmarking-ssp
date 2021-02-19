%% Script to compare MMSE and Bayesian-l1 estimators for Bernoulli-Laplace signals

K = 100;
experiment = 'deconv_gaussian';  % 'deconv_gaussian' or 'deconv_airy_disk' or 'fourier_samp'
exp_param = 4;     % parameter for the chosen experiment (see generate_datasets.m for more details)
noise_level = 10;   % input SNR
[H, M] = get_measurement_matrix(experiment, exp_param, K);

handles.Prior = 'bernoulli-laplace';      % 'student' or 'alpha-stable' or 'bernoulli-laplace'
handles.K = K;
handles.Dist_Param = [0.95, 1];

D = get_finite_difference_matrix(handles.K);

sig = generate_discrete_levy_process(handles);
pow_Hx = mean((H*sig).^2);
noise_var = pow_Hx/(10^(noise_level/10));  

meas = H*sig + sqrt(noise_var)*randn([M, 1]);

mmse_sig_params.type = 'bernoulli-laplace';
mmse_sig_params.dist_param = handles.Dist_Param;
mmse_sig_params.noise_var = noise_var;
mmse_algo_params.num_iter = 5000;
mmse_algo_params.burn_in = 2000;


[x_est_l2, x_est_l1, ux_est_l1] = bayesian_bernoulli_laplace(meas, H, D, randn([K, 1]), mmse_sig_params, mmse_algo_params);

% Plot signal and different estimates
% Plot estimates based on the quantiles of u (hinge loss)
% p = [0.1, 0.3, 0.5, 0.7, 0.9]
p = plot(ux_est_l1,'-.','LineWidth', 2); hold on; 
for ind = 1:5
    p(ind).Color(4) = 0.5;
end
% Plot posterior mean (MSE)
p = plot(x_est_l2,'--','LineWidth', 2); p.Color(4) = 0.5;
% Plot signal
p = plot(sig,'LineWidth', 2); p.Color(4) = 0.5;
legend({'$\hat{s}_{p=0.1}$', '$\hat{s}_{p=0.3}$', '$\hat{s}_{p=0.5}$', '$\hat{s}_{p=0.7}$', '$\hat{s}_{p=0.9}$', '$\hat{s}_{\mathrm{MMSE}}$', '$s$'}, 'Interpreter', 'latex', 'FontSize', 20)
