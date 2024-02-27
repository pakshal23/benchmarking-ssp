% Simulation of alpha-stable random variables

Alpha = 1.5; 
Sigma = 2; 

Variable = linspace(-10,10,4*1024);

% Direct method
StablePDF = stblpdf(Variable, Alpha, 0, Sigma, 0);
Samples = stblrnd(Alpha, 0, Sigma, 0, [10^6 1]);


figure(1);clf
plot(Variable,StablePDF),grid on, xlabel('Variable'), title ('Density')

figure(2);clf
subplot(2,1,1)
histogram(Samples,1000),grid on, xlabel('Variable'), title ('Histogram')
subplot(2,1,2)
histogram(Samples,1000000,'Normalization','pdf'),hold on, plot(Variable,StablePDF,'red')
grid on, xlim([Variable(1) Variable(end)]); xlabel('Variable'), title ('Histogram')


% Using Mixture of Gaussians
VarAux = stblrnd(Alpha/2, 1, cos(pi*Alpha/4)^(2/Alpha), 0, [10^6 1]);
SamplesBis = sqrt(2)*Sigma*sqrt(VarAux) .* randn(size(VarAux));

figure(3);clf
subplot(2,1,1)
histogram(SamplesBis,1000),grid on, xlabel('Variable'), title ('Histogram')
subplot(2,1,2)
histogram(SamplesBis,1000000,'Normalization','pdf'),hold on, plot(Variable,StablePDF,'red')
grid on, xlim([Variable(1) Variable(end)]); xlabel('Variable'), title ('Histogram')
