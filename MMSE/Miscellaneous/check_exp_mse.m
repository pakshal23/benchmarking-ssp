K = 100;
num_signals = 1000;

exp_mmse = zeros([1, num_signals]);
exp_l2 = zeros([1, num_signals]);
exp_log = zeros([1, num_signals]);

parfor i = 1:num_signals
    
    disp(i);
    
    [exp_mmse(i), exp_l2(i), exp_log(i)] = compute_exp_mse(y_cell{5,i}, H, L, x_l1{5, i, 4}, mmse_sig_params, mmse_algo_params, x_mmse{5,i}, x_l2{5,i, 4}, x_log{5,i, 5});
    
end