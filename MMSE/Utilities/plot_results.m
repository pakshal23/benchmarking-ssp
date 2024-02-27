% Plotting the results

linewid = 2;

mean_err_l1 = 10*log10(mean(best_err_l1,2));
mean_err_l2 = 10*log10(mean(best_err_l2,2));
mean_err_log = 10*log10(mean(best_err_log,2));
mean_err_mmse = 10*log10(mean(err_mmse,2));

figure;
plot(mean_err_l2, 'LineWidth', linewid, 'LineStyle', '-', 'Marker', 'o');
hold on
plot(mean_err_l1, 'LineWidth', linewid, 'LineStyle', '--', 'Marker', '+');
hold on
plot(mean_err_log, 'LineWidth', linewid, 'LineStyle', ':', 'Marker', 's');
hold on
plot(mean_err_mmse, 'LineWidth', linewid, 'LineStyle', '-.', 'Marker', 'd');

grid on
legend('l2', 'l1', 'log', 'mmse');
title([experiment, ' - ', 'Experiment parameter: ', num2str(exp_param), ' , ', 'noise level: ', num2str(noise_level)]);
xlabel('Signal parameter');
ylabel('MSE (in dB)');
%xticklabels(sig_params);
xticklabels([20, 10, 5, 2, 1.5, 1.4, 1]);