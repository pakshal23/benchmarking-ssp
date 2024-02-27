%% Script to test the MMSE estimator for Bernoulli-Laplace signals by comparing with a message-passing 
%  based MMSE estimator for denoising

parpool(4);

%% Setting
sig_type = 'bernoulli-laplace';
noise_level = 30;   % input SNR

K = 100;            % Dimension of the signal
num_signals = 50;
M = K;
H = eye(K);         % Identity forward model (denoising)

num_samp_sig = 5000;   % We use num_samp_sig temporary signals to determine the noise variance from the noise level (input SNR)

handles.Prior = 'bernoulli-laplace';
handles.K = K;
handles.Dist_Param = [0.8, 1];

D = get_finite_difference_matrix(handles.K);

%% Determining the noise variance using sample signals
pow_x = zeros([num_samp_sig, 1]);
for j = 1:num_samp_sig
    temp_sig = generate_discrete_levy_process(handles);
    pow_x(j, 1) = mean((H*temp_sig).^2);
end
med_pow_x = median(pow_x);
noise_var = med_pow_x/(10^(noise_level/10));

%% Setting up the optimization parameters and estimators
u0 = zeros([handles.K,1]);  % initialization

% Parameters for the Gibbs sampling based estimator
mmse_sig_params.type = sig_type;
mmse_sig_params.dist_param = handles.Dist_Param;
mmse_sig_params.noise_var = noise_var;
mmse_algo_params.num_iter = 7500;
mmse_algo_params.burn_in = 2500;

% Parameters for the message passing based estimator
mp_params.Prior_Type = 2;
mp_params.Dist_Sel = 4;
mp_params.MassProb = handles.Dist_Param(1);
mp_params.Dist_Param = handles.Dist_Param(2);
mp_params.Noise_Var = noise_var;

Hval = parallel.pool.Constant(H);
Lval = parallel.pool.Constant(L);
u0val = parallel.pool.Constant(u0);

%% Structures to store the computed results
x_cell = cell([1, num_signals]);
y_cell = cell([1, num_signals]);

x_mmse = cell([1, num_signals]);
err_mmse = zeros([1, num_signals]);

x_mp = cell([1, num_signals]);
err_mp = zeros([1, num_signals]);

%% Generating the signals
for j = 1:num_signals
    x_cell{1, j} = generate_discrete_levy_process(handles);
    y_cell{1, j} = x_cell{1, j} + sqrt(noise_var)*randn([M, 1]);
end

%% Running the estimators
parfor j = 1:num_signals
    disp(['Signal Number: ', num2str(j)]);

    sig = x_cell{1, j};
    meas = y_cell{1, j};
    
    % Message-passing MMSE estimator for denoising
    x_mp{j} = denoiseMpMmse(meas, mp_params);
    err_mp(j) = norm(sig - x_mp{j})^2;
    
    % stable mmse-estimator
    x_mmse{j} = gs_mmse_estimator(meas, Hval.Value, Lval.Value, u0val.Value, mmse_sig_params, mmse_algo_params);
    err_mmse(j) = norm(sig - x_mmse{j})^2;
end
    