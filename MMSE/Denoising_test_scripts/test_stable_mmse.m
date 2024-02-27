%% Script to test the Gibbs MMSE for alpha-stable signals by comparing to the message-passing based MMSE for denoising (Kamilov 2013)

% Configure plots
plt = false;

% Set seed for consistent profiling
seed = 200;
rng(seed);

%% Settings
% Parallelize generation of auxiliary variables in Gibbs sampler - Using
% parfor in sample_aux_variable seemed to slow down performance, does not
% make sense
%if isempty(gcp('nocreate'))
%    parpool();
%end
% Type of signal
sig_type = 'alpha-stable';
% Input SNR - Sketchy, signal has infinite power
noise_level = 30;
% Signal dimension
K = 100;
% Number of signals
num_signals = 1;
% Forward model (identity -> denoising)
H = eye(K);
% Parameters for generation of Levy processes
genlevy_params.Prior = 'alpha-stable';
genlevy_params.K = K;
genlevy_params.Dist_Param = [1, 0, 1, 0];
% Define the finite difference matrix
D = get_finite_difference_matrix(genlevy_params.K);

%% Sketchy way to set noise variance, but easier than making it up
% Number of sampling signals to use to determine the noise level
num_samp_sig = 5000;
% Determining the noise variance using sample signals
pow_x = zeros([num_samp_sig, 1]);
for j = 1:num_samp_sig
    temp_sig = generate_discrete_levy_process(genlevy_params);
    pow_x(j, 1) = mean((H*temp_sig).^2);
end
med_pow_x = median(pow_x);
noise_var = med_pow_x/(10^(noise_level/10));

%% Setting up the optimization parameters and estimators
% Parameters for the Gibbs sampling based estimator
mmse_sig_params.type = sig_type;
mmse_sig_params.dist_param = genlevy_params.Dist_Param;
mmse_sig_params.noise_var = noise_var;
mmse_algo_params.num_iter = 7500;
mmse_algo_params.burn_in = 2500;

% Parameters for the message passing based estimator
u0 = zeros([genlevy_params.K,1]);
mp_params.Prior_Type = 1;
mp_params.Dist_Sel = 4;
mp_params.Dist_Param = genlevy_params.Dist_Param(1);
mp_params.Noise_Var = noise_var;

% Shared constant values among workers
%Hval = parallel.pool.Constant(H);
%Lval = parallel.pool.Constant(D);
%u0val = parallel.pool.Constant(u0);
Hval.Value = H; Lval.Value = D; u0val.Value = u0;


%% Structures to store the computed results
x_cell = cell([1, num_signals]);
y_cell = cell([1, num_signals]);

x_mmse = cell([1, num_signals]);
err_mmse = zeros([1, num_signals]);

x_mp = cell([1, num_signals]);
err_mp = zeros([1, num_signals]);

%% Generating the signals
for j = 1:num_signals
    x_cell{1, j} = generate_discrete_levy_process(genlevy_params);
    y_cell{1, j} = x_cell{1, j} + sqrt(noise_var)*randn([K, 1]);
end

%% Running the estimators
fprintf('\nMessage passing\n\tSignal')
tic;
for j = 1:num_signals    
    % Let user know the program is doing stuff
    fprintf(" %d,", j);

    % Extract corresponding signal and measurements (slicing)
    sig = x_cell{1, j};
    meas = y_cell{1, j};
    
    % Compute message-passing MMSE estimator for denoising with an
    % alpha-stable prior
    x_mp{j} = denoiseMpMmse(meas, mp_params);
    err_mp(j) = norm(sig - x_mp{j})^2/K;
    
end
fprintf(" Done.")
t_mp = toc;

fprintf('\n\tSummary: Completed %d estimations in %.3f s, obtaining an RMSE of %.3E.\n', num_signals, t_mp, sqrt(mean(err_mp)) )

if plt
    % Plot best estimate
    % Recover it
    [errmin,indmin] = min(err_mp);
    sig = x_cell{1,indmin};
    est_mp = x_mp{indmin};
    % Plot
    figure(1); clf(1)
    plot(sig); hold on; plot(est_mp); legend('Signal', 'Estimated signal')
    title(sprintf('Message passing - Best estimate, RMSE = %.3E', sqrt(errmin))) 
    drawnow;
end

fprintf('\nGibbs sampling\n\tSignal')
tic;
%parfor j = 1:num_signals
for j = 1:num_signals
    
    % Let user know the programis doing stuff
    fprintf(" %d,", j);

    % Extract corresponding signal and measurements (slicing)
    sig = x_cell{1, j};
    meas = y_cell{1, j};

    % Compute Gibbs sampling MMSE estimator (posterior mean)
    x_mmse{j} = gs_mmse_estimator(meas, Hval.Value, Lval.Value, [], mmse_sig_params, mmse_algo_params);
    err_mmse(j) = norm(sig - x_mmse{j})^2/K;
end
fprintf(" Done.")
t_mmse = toc / num_signals;

fprintf('\n\tSummary: Completed %d estimations in %.3f s, obtaining an RMSE of %.3E.\n', num_signals, t_mmse, sqrt(mean(err_mmse)) )

if plt
    % Plot best estimate
    % Recover it
    [errmin,indmin] = min(err_mmse);
    sig = x_cell{1,indmin};
    est_mmse = x_mmse{indmin};
    % Plot
    figure(2); clf(2)
    plot(sig); hold on; plot(est_mmse); 
    title(sprintf('Gibbs sampling - Best estimate, RMSE = %.3E', sqrt(errmin))) 
    drawnow;
end

fprintf('\nComparison\n\tGibbs took %.2f times longer than message passing and RMSE_Gibbs / RMSE_MP = %.2f.\n\n', t_mmse/t_mp, sqrt(mean(err_mmse))/sqrt(mean(err_mp)) )
