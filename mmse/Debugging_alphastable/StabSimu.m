% StabSim.m -- JFG -- 05/07/2020 -- Simul de Alpha-Stable
%

% Nettoyage and Co
    clear all
    %close all

% Définition des paramètres 
    Alpha = 1.25; % Dans ] 0 , 2 ]
    Sigma = 2; % Réel positif
    %
    Variable = linspace(-10,10,4*1024);

% Définition de la forme, densité, tirages...
    Stable = makedist('Stable','alpha',Alpha,'beta',0,'gam', Sigma,'delta',0);
    StablePDF = pdf(Stable, Variable);
    Samples = random(Stable,[ 10^6 1 ]);    

% Tracé divers
    figure(1);clf
    plot(Variable,StablePDF),grid on, xlabel('Variable'), title ('Density')
    %
    figure(2);clf
    subplot(2,1,1)
        histogram(Samples,1000),grid on, xlabel('Variable'), title ('Histogram')
    subplot(2,1,2)
        histogram(Samples,1000000,'Normalization','pdf'),hold on, plot(Variable,StablePDF,'red')
        grid on, xlim([Variable(1) Variable(end)]); xlabel('Variable'), title ('Histogram')

% Re-simul par Mixture of Gaussian
    StableAux = makedist('Stable','alpha',Alpha/2,'beta',1,'gam', cos(pi*Alpha/4)^(2/Alpha),'delta',0);
    VarAux = random(StableAux,[ 10^7 1 ]);
    %VarAux = VarAux( VarAux>0 );
    %VarAux = abs(VarAux);
    SamplesBis = sqrt(2)*Sigma*sqrt(VarAux) .* randn(size(VarAux));
    
% Tracé divers
    figure(3);clf
    subplot(2,1,1)
        histogram(SamplesBis,1000),grid on, xlabel('Variable'), title ('Histogram')
    subplot(2,1,2)
        histogram(SamplesBis,1000000,'Normalization','pdf'),hold on, plot(Variable,StablePDF,'red')
        grid on, xlim([Variable(1) Variable(end)]); xlabel('Variable'), title ('Histogram')
        
