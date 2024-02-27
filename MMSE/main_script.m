%% Main script for comparing l1, l2, log and MMSE estimators in a deconvolution or Fourier sampling task.

parpool(4);
warning('off');
pctRunOnAll warning off

%% Settings
data_folder = '/Users/bohra/Desktop/BenchmarkingSSP/mmse/deconvolution/data';
experiment = 'deconv_gaussian';     % 'deconv_gaussian' or 'deconv_airy_disk' or 'fourier_samp'
sig_type = 'bernoulli-laplace';       % 'student' or 'bernoulli-laplace'
data_label = 'test';            % 'train' or 'valid' or 'test'

exp_param = 4;     % parameter for the chosen experiment (see generate_datasets.m for more details)
noise_level = 30;   % input SNR
sig_param_idx = 1;  % Index indicating the signal parameter (see generate_datasets.m for more details)

save_dir = '/Users/bohra/Desktop/BenchmarkingSSP/mmse/deconvolution/results';

%% Loading the data and setting up the directory to store the results
if (strcmp(sig_type, 'student'))
    sig_string = 'student';
elseif (strcmp(sig_type, 'bernoulli-laplace'))
    sig_string = 'bl';
end

filename = [data_folder, '/', experiment, '_', num2str(exp_param), '_', sig_string, '_', 'sigparamidx', '_', num2str(sig_param_idx), '_', 'noiselevel', '_', num2str(noise_level), '_', data_label];
load(filename);

resultname = [save_dir, '/', experiment, '_', num2str(exp_param), '_', sig_string, '_', 'sigparamidx', '_', num2str(sig_param_idx), '_', 'noiselevel', '_', num2str(noise_level), '_',  data_label, '_', 'results'];

%% Setting up the optimization parameters and estimators
% Parameters for the l1 and log estimators (iterative schemes)
opt_params.name = 'admm';      % 'fista' or 'admm'
opt_params.maxiter = 1e5;
opt_params.relative_tol = 1e-8;
opt_params.verbose = 0;
opt_params.ItUpOut = 10;
opt_params.iterVerb = 10;

x0 = zeros([handles.K,1]);  % initialization for the iterative schemes
lambda_list = 10.^(-6 : 0.5 : 3);   % Range of lambda values to tune over (for l1 and log estimators)
num_lambda = length(lambda_list);
lambda_list_l2 = 10.^(-8 : 0.5 : 3);   % Range of lambda values to tune over (for l1 and log estimators)
num_lambda_l2 = length(lambda_list_l2);

% Parameters for the MMSE estimator
mmse_sig_params.type = sig_type;
mmse_sig_params.dist_param = sig_params{sig_param_idx};
mmse_sig_params.noise_var = noise_var(sig_param_idx);
mmse_algo_params.num_iter = 3000;
mmse_algo_params.burn_in = 1000;

% Defining function handles for the l1, l2 and log estimators (to pass to the tune_lambda routine) 
l2_estimator_fun = @(y_val, lambda_val, params, x0_val) l2_estimator(y_val, H, H'*H, D, D'*D, lambda_val, params, x0_val);
l1_estimator_fun = @(y_val, lambda_val, params, x0_val) l1_estimator(y_val, H, D, lambda_val, params, x0_val);
log_estimator_fun = @(y_val, lambda_val, params, x0_val) log_estimator(y_val, H, D, 1, lambda_val, params, x0_val);

% 
Hval = parallel.pool.Constant(H);
Dval = parallel.pool.Constant(D);
x0val = parallel.pool.Constant(x0);

%% Structures to store the computed results
num_signals = size(y_cell, 2);

x_mmse = cell([1, num_signals]);
err_mmse = zeros([1, num_signals]);

x_l2 = cell([num_signals, num_lambda_l2]);
err_l2 = zeros([num_signals, num_lambda_l2]);
opt_cost_l2 = zeros([num_signals, num_lambda_l2]);
num_iter_l2 = zeros([num_signals, num_lambda_l2]);

x_l1 = cell([num_signals, num_lambda]);
err_l1 = zeros([num_signals, num_lambda]);
opt_cost_l1 = zeros([num_signals, num_lambda]);
num_iter_l1 = zeros([num_signals, num_lambda]);

x_log = cell([num_signals, num_lambda]);
err_log = zeros([num_signals, num_lambda]);
opt_cost_log = zeros([num_signals, num_lambda]);
num_iter_log = zeros([num_signals, num_lambda]);

%% Running the l1 and l2 estimators    
parfor j = 1:num_signals
    disp(['Signal Number: ', num2str(j)]);

    sig = x_cell{1, j};
    meas = y_cell{1, j};

    % l2-estimator
    [x_l2(j,:), err_l2(j,:), opt_cost_l2(j,:), num_iter_l2(j,:)] = tune_lambda(l2_estimator_fun, meas, sig, lambda_list_l2, [], opt_params, x0val.Value);

    % l1-estimator
    [x_l1(j,:), err_l1(j,:), opt_cost_l1(j,:), num_iter_l1(j,:)] = tune_lambda(l1_estimator_fun, meas, sig, lambda_list, [], opt_params, x0val.Value);
end
save(resultname);

%% Choosing the best lambda for the l1 and l2 estimators
best_err_l2 = zeros([1, num_signals]);
best_err_l1 = zeros([1, num_signals]);
avg_err_l2 = zeros([num_lambda_l2,1]);
avg_err_l1 = zeros([num_lambda,1]);
x0_log = cell([1, num_signals]); % We take the best l1 results to initialize the log estimator

for j = 1:num_lambda_l2
    avg_err_l2(j,1) = mean(err_l2(:, j));
end

for j = 1:num_lambda
    avg_err_l1(j,1) = mean(err_l1(:, j));
end
    
[min_avg_err_l2, min_ind] = min(avg_err_l2);
min_ind_l2 = min_ind(1);
    
[min_avg_err_l1, min_ind] = min(avg_err_l1);
min_ind_l1 = min_ind(1);

best_err_l2(1, :) = err_l2(:, min_ind_l2);
best_err_l1(1, :) = err_l1(:, min_ind_l1);
x0_log(1, :) = x_l1(:, min_ind_l1);
    
save(resultname);

%% Running the log and MMSE estimators
parfor j = 1:num_signals
    disp(['Signal Number: ', num2str(j)]);

    sig = x_cell{1, j};
    meas = y_cell{1, j};
    curr_x0_log = x0_log{1, j};
    %curr_x0_log  = randn(100,1);

    % log-estimator
    [x_log(j,:), err_log(j,:), opt_cost_log(j,:), num_iter_log(j,:)] = tune_lambda(log_estimator_fun, meas, sig, lambda_list, [], opt_params, curr_x0_log);

    % MMSE-estimator
    %x_mmse{j} = gs_mmse_estimator(meas, Hval.Value, Dval.Value, x0val.Value, mmse_sig_params, mmse_algo_params);
    x_mmse{j} = gs_mmse_estimator(meas, Hval.Value, Dval.Value, (Dval.Value)*curr_x0_log, mmse_sig_params, mmse_algo_params);
    err_mmse(j) = norm(sig - x_mmse{j})^2;
end
save(resultname);
    
%% Choosing the best lambda for the log estimator
best_err_log = zeros([1, num_signals]);
avg_err_log = zeros([num_lambda, 1]);
    
for j = 1:num_lambda
    avg_err_log(j,1) = mean(err_log(:, j));
end
    
[min_avg_err_log, min_ind] = min(avg_err_log);
min_ind_log = min_ind(1);
best_err_log(1, :) = err_log(:, min_ind_log);

save(resultname);