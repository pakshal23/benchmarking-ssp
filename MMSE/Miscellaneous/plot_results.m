linewid = 2;

mean_snr_l2 = zeros([num_sig_params, 1]);
mean_snr_l1 = zeros([num_sig_params, 1]);
mean_snr_mmse = zeros([num_sig_params, 1]);
mean_snr_cnn = zeros([num_sig_params, 1]);

for i = 1:num_sig_params
   
    total_pow = 0;
    for j = 1:num_signals
        total_pow = total_pow + norm(x_cell{i, j})^2;
    end
    
    mean_snr_l2(i,1) = 10*log10(total_pow/num_signals) - 10*log10(mean(best_err_l2(i,:)));
    mean_snr_l1(i,1) = 10*log10(total_pow/num_signals) - 10*log10(mean(best_err_l1(i,:)));
    %mean_snr_mmse(i,1) = 10*log10(total_pow/num_signals) - 10*log10(mean(err_mmse(i,:)));
    %mean_snr_cnn(i,1) = 10*log10(total_pow/num_signals) - 10*log10(mean_err_cnn(i));
    
end

figure;
plot(mean_snr_l2, 'LineWidth', linewid, 'LineStyle', '-', 'Marker', 'o');
hold on
plot(mean_snr_l1, 'LineWidth', linewid, 'LineStyle', '--', 'Marker', 'square');
% hold on
%plot(mean_snr_mmse, 'LineWidth', linewid, 'LineStyle', '--', 'Marker', 'd');
%hold on
%plot(mean_snr_cnn, 'LineWidth', linewid, 'LineStyle', '--', 'Marker', 'x');


grid on
%legend('l2', 'l1', 'mmse', 'CNN');
xlabel('Signal Parameter');
ylabel('SNR (in dB)');