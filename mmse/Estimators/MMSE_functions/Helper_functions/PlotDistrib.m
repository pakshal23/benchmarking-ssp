% PlotDistrib -- Pakshal & J.-F. -- June 2020

% Clear and define parameters
    clear all
    Alpha = 1;
    u = 1000;

% Grid of values
    Lambda = linspace(1e4,5*1e6,1e3);

% Fist factor (from Gaussian part) 
    Fact1 = exp( - u^2 ./ (4*Lambda) ) ./ sqrt(4*pi*Lambda) ; 

% Second factor (from Alpha-Stable part)
    StablePDF = makedist('Stable','Alpha',Alpha/2,'beta',1,'gam', (cos(pi*Alpha/4))^(2/Alpha),'delta',0);
    Fact2 = pdf(StablePDF, Lambda);

% Density
    CondPDF = Fact1 .* Fact2;

% Maximum and argument
    ArgMax = u^2 / 2;
    Max = 1 / sqrt(2*pi*u^2*exp(1));

% Plots
    figure(1);clf
    plot(Lambda,Fact1,'red');hold on;plot(ArgMax,Max,'*k'),plot(Lambda,Fact2,'blue')
    grid on, xlabel('Variable Lambda'), title ('Facteurs: red:FomGauss -- blue:FromStable')

    figure(2);clf
    subplot(211)
        plot(Lambda,CondPDF,'green')
        grid on, xlabel('Variable Lambda'), title ('Density (green)')
     subplot(212)
        plot(Lambda,CondPDF,'green');hold on;plot(Lambda,Max * Fact2,'black')
        grid on, xlabel('Variable Lambda'), title ('Density (green) and majorant (black)')

   
% And sampling
%     for Index = 1:100
%     Samples(Index) = SampleViaCDF(CondPDF,Lambda);
%     end
% 
%     figure(3);clf
%     subplot(211)
%         plot(Lambda,CondPDF,'green')
%         grid on; xlabel('Variable Lambda'), title ('Density (green)')
%      subplot(212)
%         hist(Samples,100)
%         grid on; xlabel('Variable Lambda'), title ('Histogram')


