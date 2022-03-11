close all; clear all; clc

nrof_samples = 10000;
%% Target distribution parameters
% Changeable
alpha = 1.5;
gamma = 1;
% Fixed
% Symmetric
beta = 0;
% No shift
delta = 0;

%% Mix parameters
% New alpha and gamma
alpha_mix = alpha/2;
gamma_mix = cos(pi*alpha/4)^(2/alpha);
% Positive mixture of Gaussians
beta_mix = 1;
% No shift
delta_mix = 0;

%% Sample
% Sample from target distributions
true_samples = generate_alpha_stable_rv([alpha, beta, gamma, delta], nrof_samples);
% Sample from the mixture
lambda_values = generate_alpha_stable_rv([alpha_mix, beta_mix, gamma_mix, delta_mix], nrof_samples);
mix_samples = gamma*sqrt(2*lambda_values).*randn(nrof_samples,1);

%% Compare
figure; 
mi = min(min(true_samples),min(mix_samples));
ma = max(max(true_samples),max(mix_samples));
bin_centers_cdf = mi:(ma-mi)/1000:ma;
bin_centers_hist = mi:(ma-mi)/100:ma;
subplot(1,2,1)
histogram(true_samples,bin_centers_hist)
h1 = hist(true_samples,bin_centers_cdf); hold on; 
h2 = hist(mix_samples,bin_centers_cdf);
histogram(mix_samples,bin_centers_hist)
legend('True samples', 'Samples from the mixture')
cdf1 = cumsum(h1)/nrof_samples;
cdf2 = cumsum(h2)/nrof_samples;
subplot(1,2,2)
plot(bin_centers_cdf, cdf1, 'b', 'LineWidth', 2, 'DisplayName', 'Direct'); hold on;
plot(bin_centers_cdf, cdf2, 'r-.', 'LineWidth', 2, 'DisplayName', 'Mixture')
legend('True samples', 'Samples from the mixture')