close all, clear all, clc

alpha = 1;
thresholds = stblinv([0.001,0.99], alpha/2, 1, (cos(pi*alpha/4))^(2/alpha), 0);
% Aligned distributions
plot_distributions(alpha, geomean(thresholds))
f = gcf;
figure(f)
title(strcat("Aligned distributions (Normalized to max 1), $\alpha= $",...
      sprintf("%.2f", alpha), " $u= $", sprintf("%.2f", mean(thresholds))),...
      'Interpreter', 'latex', 'FontSize', 15);
% Data much larger than prior
plot_distributions(alpha,thresholds(2))
f = gcf;
figure(f)
% IG((alpha+1)/2, u^2/2)
lamb = logspace(-6,6,400);
a = (alpha+1)/2; b = thresholds(2)^2/2;
pdf = b^a/gamma(a) * lamb.^(-a-1).*exp(-b./lamb);
pdf = pdf/max(pdf);
semilogx(lamb, pdf, 'k', 'LineWidth', 2, 'DisplayName', '$IG((\alpha+1)/2,u^2/2)$');
title(strcat("Data $>>$ Prior - Assymptotic IG approx (Normalized to max 1), $\alpha= $",...
       sprintf("%.2f", alpha), " $u= $", sprintf("%.2f", thresholds(2))),...
       'Interpreter', 'latex', 'FontSize', 15);
% Data much smaller than prior
plot_distributions(alpha,thresholds(1))
f = gcf;
figure(f)
title(strcat("Data $<<$ Prior - Truncation (Normalized to max 1), $\alpha= $",...
       sprintf("%.2f", alpha), " $u= $", sprintf("%.2f", thresholds(1))),...
       'Interpreter', 'latex', 'FontSize', 15);
close 1 3 5