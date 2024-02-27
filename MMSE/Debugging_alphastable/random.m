%% Time standard implementation

mmse_stable_algo_params.mode = 'standard';

tic;
for j = 1:num_signals

    disp(['Signal Number: ', num2str(j)]);

    sig = x_cell{1, j};
    meas = y_cell{1, j};
    
    % Message-passing MMSE estimator for denoising
    x_mp{j} = denoiseMpMmse(meas,mp_params);
    err_mp(j) = norm(sig - x_mp{j})^2;
    
    % stable mmse-estimator
    %[x_mmse_stable{j}, exp_mse_stable(j)] = gs_mmse_estimator(meas, Hval.Value, Lval.Value, initval.Value, mmse_stable_sig_params, mmse_stable_algo_params);
    %x_mmse_stable{j} = gs_mmse_estimator(meas, H, L, randn([K,1]), mmse_stable_sig_params, mmse_stable_algo_params);
    x_mmse_stable{j} = gs_mmse_estimator(meas, H, L, c0, mmse_stable_sig_params, mmse_stable_algo_params);
    err_mmse_stable(j) = norm(sig - x_mmse_stable{j})^2;
    
%     % student mmse-estimator
    %[x_mmse_student{j}, exp_mse_student(j)] = gs_mmse_estimator(meas, Hval.Value, Lval.Value, initval.Value, mmse_student_sig_params, mmse_student_algo_params);
    %x_mmse_student{j} = gs_mmse_estimator(meas, H, L, randn([K,1]), mmse_student_sig_params, mmse_student_algo_params);
    %err_mmse_student(j) = norm(sig - x_mmse_student{j})^2;

end
t1 = toc;
mean(err_mmse_stable)

% %% Time slice implementation
% 
% mmse_stable_algo_params.mode = 'slice';
% 
% tic;
% for j = 1:num_signals
% 
%     disp(['Signal Number: ', num2str(j)]);
% 
%     sig = x_cell{1, j};
%     meas = y_cell{1, j};
%     
%     % Message-passing MMSE estimator for denoising
%     %x_mp{j} = denoiseMpMmse(meas,mp_params);
%     %err_mp(j) = norm(sig - x_mp{j})^2;
%     
%     % stable mmse-estimator
%     %[x_mmse_stable{j}, exp_mse_stable(j)] = gs_mmse_estimator(meas, Hval.Value, Lval.Value, initval.Value, mmse_stable_sig_params, mmse_stable_algo_params);
%     %x_mmse_stable{j} = gs_mmse_estimator(meas, H, L, randn([K,1]), mmse_stable_sig_params, mmse_stable_algo_params);
%     %x_mmse_stable{j} = gs_mmse_estimator(meas, H, L, c0, mmse_stable_sig_params, mmse_stable_algo_params);
%     x_mmse_stable{j} = gs_mmse_estimator(meas, H, L, L*sig, mmse_stable_sig_params, mmse_stable_algo_params);
%     err_mmse_stable(j) = norm(sig - x_mmse_stable{j})^2;
%     
% %     % student mmse-estimator
%     %[x_mmse_student{j}, exp_mse_student(j)] = gs_mmse_estimator(meas, Hval.Value, Lval.Value, initval.Value, mmse_student_sig_params, mmse_student_algo_params);
%     x_mmse_student{j} = gs_mmse_estimator(meas, H, L, c0, mmse_student_sig_params, mmse_student_algo_params);
%     err_mmse_student(j) = norm(sig - x_mmse_student{j})^2;
% 
% end
% t2 = toc;