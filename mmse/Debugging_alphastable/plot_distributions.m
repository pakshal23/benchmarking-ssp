function [] = plot_distributions(alpha, u)

lamb = (0.01:0.01:5);

term1 = (1./(sqrt(4*pi*lamb))).*exp(-(u^2)./(4*lamb));

stable_pdf = makedist('Stable','alpha',alpha/2,'beta',1,'gam', (cos(pi*alpha/4))^(2/alpha),'delta',0);
term2 = pdf(stable_pdf, lamb);

conditional_pdf = term1.*term2;

C = (1/sqrt(2*pi))*(1/abs(u))*exp(-0.5);

figure;
plot(conditional_pdf);
hold on;
plot(C.*term2);

end

