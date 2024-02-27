%% This script generates datasets (training, validation and testing) and stores them in some directory


%%
save_data = 0;
dest_folder = '/Users/bohra/Desktop/BenchmarkingSSP/mmse/deconvolution/data';
experiment = 'deconv_gaussian';  % 'deconv_gaussian' or 'deconv_airy_disk' or 'fourier_samp'

K = 100;
num_train_signals = 1000;
num_valid_signals = 1000;
num_test_signals = 1000;

num_samp_sig = 5000;   % We use num_samp_sig temporary signals to determine the noise variance from the noise level (blur SNR)

handles.Prior = 'bernoulli-laplace';      % 'student' or 'bernoulli-laplace'
handles.K = K;

D = get_finite_difference_matrix(handles.K);


%% Signal parameters depending on the type of the signal
if (strcmp(handles.Prior, 'student'))
    
    sig_params = {20, 10, 5, 2, 1.5, 1.4, 1.3, 1.2, 1.1, 1.05, 1};
    num_sig_params =length(sig_params);
    sig_string = 'student';
        
elseif (strcmp(handles.Prior, 'bernoulli-laplace'))
    
    b = 1;
    sig_params = {[0.6, b], [0.7, b], [0.8, b], [0.9, b]};
    num_sig_params =length(sig_params);
    sig_string = 'bl';
    
end


%% Experiment parameters depending on the type of experiment
if (strcmp(experiment, 'deconv_gaussian'))
    
    noise_level = 20;                                                      % Noise level (input SNR) for the additive noise
    exp_param = 4;                                                         % variance of the gaussian blur / filter
    [H, M] = get_measurement_matrix(experiment, exp_param, K);
    H = eye(K); M = K;
    
elseif (strcmp(experiment, 'deconv_airy_disk'))
    
    noise_level = 30;                                                      % Noise level (input SNR) for the additive noise
    exp_param = 31;                                                        % Filter size
    [H, M] = get_measurement_matrix(experiment, exp_param, K);
    
elseif (strcmp(experiment, 'fourier_samp'))
    
    noise_level = 30;                                                      % Noise level (input SNR) for the additive noise
    exp_param = 31;                                                        % Number of fourier samples
    [H, M] = get_measurement_matrix(experiment, exp_param, K);

end
    

%% Determining the noise variance using sample signals
pow_Hx = zeros([num_sig_params, num_samp_sig]);
for i = 1:num_sig_params

    handles.Dist_Param = sig_params{i};

    for j = 1:num_samp_sig

        temp_sig = generate_discrete_levy_process(handles);
        %pow_Hx(i, j) = var(H*temp_sig);
        pow_Hx(i, j) = mean((H*temp_sig).^2);
        
    end
end
med_pow_Hx = median(pow_Hx, 2);


%% To store the data 
x_cell_train = cell([num_sig_params, num_train_signals]);
y_cell_train = cell([num_sig_params, num_train_signals]);
            
x_cell_valid = cell([num_sig_params, num_valid_signals]);            
y_cell_valid = cell([num_sig_params, num_valid_signals]);
   
x_cell_test = cell([num_sig_params, num_test_signals]);
y_cell_test = cell([num_sig_params, num_test_signals]);            
            
            
%% Generating the datasets
noise_var = med_pow_Hx./(10^(noise_level/10));                 
for i = 1:num_sig_params

    handles.Dist_Param = sig_params{i};
    var_n = noise_var(i);
    
    for j = 1:num_train_signals

        x_cell_train{i, j} = generate_discrete_levy_process(handles);
        y_cell_train{i, j} = H*x_cell_train{i, j} + sqrt(var_n)*randn([M, 1]);

    end
    
    for j = 1:num_valid_signals

        x_cell_valid{i, j} = generate_discrete_levy_process(handles);
        y_cell_valid{i, j} = H*x_cell_valid{i, j} + sqrt(var_n)*randn([M, 1]);

    end
    
    for j = 1:num_test_signals

        x_cell_test{i, j} = generate_discrete_levy_process(handles);
        y_cell_test{i, j} = H*x_cell_test{i, j} + sqrt(var_n)*randn([M, 1]);

    end
end


%% Saving datasets (separately per signal parameter)

if (save_data)
    
    for i = 1:num_sig_params
       
        filename = [dest_folder, '/', experiment, '_', num2str(exp_param), '_', sig_string, '_', 'sigparamidx', '_', num2str(i), '_', 'noiselevel', '_', num2str(noise_level), '_', 'train'];
        x_cell = x_cell_train(i,:); y_cell = y_cell_train(i,:);
        save(filename, 'y_cell', 'H', 'D', 'x_cell', 'experiment', 'handles', 'noise_level', 'exp_param', 'sig_params', 'noise_var', '-v7');

        filename = [dest_folder, '/', experiment, '_', num2str(exp_param), '_', sig_string, '_', 'sigparamidx', '_', num2str(i), '_', 'noiselevel', '_', num2str(noise_level), '_', 'valid'];
        x_cell = x_cell_valid(i,:); y_cell = y_cell_valid(i,:);
        save(filename, 'y_cell', 'H', 'D', 'x_cell', 'experiment', 'handles', 'noise_level', 'exp_param', 'sig_params', 'noise_var', '-v7');

        filename = [dest_folder, '/', experiment, '_', num2str(exp_param), '_', sig_string, '_', 'sigparamidx', '_', num2str(i), '_', 'noiselevel', '_', num2str(noise_level), '_', 'test'];
        x_cell = x_cell_test(i,:); y_cell = y_cell_test(i,:);
        save(filename, 'y_cell', 'H', 'D', 'x_cell', 'experiment', 'handles', 'noise_level', 'exp_param', 'sig_params', 'noise_var', '-v7');
        
    end
end
      

%% Saving the datasets together

% if (save_data)
%     
%     filename = [dest_folder, experiment, '_', num2str(exp_param), '_', num2str(noise_level), '_', sig_string, '_', 'train'];
%     x_cell = x_cell_train; y_cell = y_cell_train;
%     save(filename, 'y_cell', 'H', 'L', 'x_cell', 'experiment', 'handles', 'noise_level', 'exp_param', 'sig_params', 'noise_var', '-v7.3');
%     
%     filename = [dest_folder, experiment, '_', num2str(exp_param), '_', num2str(noise_level), '_', sig_string, '_', 'valid'];
%     x_cell = x_cell_valid; y_cell = y_cell_valid;
%     save(filename, 'y_cell', 'H', 'L', 'x_cell', 'experiment', 'handles', 'noise_level', 'exp_param', 'sig_params', 'noise_var', '-v7.3');
%     
%     filename = [dest_folder, experiment, '_', num2str(exp_param), '_', num2str(noise_level), '_', sig_string, '_', 'test'];
%     x_cell = x_cell_test; y_cell = y_cell_test;
%     save(filename, 'y_cell', 'H', 'L', 'x_cell', 'experiment', 'handles', 'noise_level', 'exp_param', 'sig_params', 'noise_var', '-v7.3');
%       
% end