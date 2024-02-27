function [] = plot_distributions(alpha, u, maxlam)
if nargin < 3
    maxlam = 1000;
    if nargin < 2
        u = 0.5;
        if nargin < 1
            alpha = 1;
        end
    end
end

    % \lambda axis
    lamb = logspace(-6,6,400);
    
    % Distribution we want to sample from 
    % Term arising from Gaussian mixture
    term1 = (1 ./ sqrt(4*pi*lamb) ) .* exp( -(u^2) ./ (4*lamb) );
    % \alpha^'-stable term
    term2 = stblpdf(lamb, alpha/2, 1, (cos(pi*alpha/4))^(2/alpha), 0);
    thresholds = stblinv([0.001,0.99], alpha/2, 1, (cos(pi*alpha/4))^(2/alpha), 0);
    % Distribution
    conditional_pdf = term1.*term2;

    % Constant C for rejection sampling? 
    C = (1/sqrt(2*pi)) * (1/abs(u)) * exp(-0.5);

    figure;
        subplot(1, 2, 1)
            plot( lamb, conditional_pdf, 'LineWidth', 2 );
            title("$p(\lambda | u, \tau,y)$", ...
                    'Interpreter', 'latex', 'FontSize', 15)
            xline(thresholds(1));xline(thresholds(2))
        subplot(2, 2, 2)
            plot( lamb, term2, 'LineWidth', 2 ); % Could multiply by C for comparable height
            xline(thresholds(1));xline(thresholds(2))
            title("$f_{\alpha/2}(\lambda;1,\cos(\pi\alpha/4)^{2/\alpha},0)$", ...
               'Interpreter', 'latex', 'FontSize', 15)
        subplot(2, 2, 4)
            plot( lamb, term1, 'LineWidth', 2 );
            hold on;
            plot( [min(lamb), max(lamb)], C * ones(2,1), 'r', 'LineWidth', 2 )
            title("$\mathcal{N}(u_k; 0, 2 c^2 \lambda_k)$", ...
               'Interpreter', 'latex', 'FontSize', 15)
            legend("$\mathcal{N}(u_k; 0, 2 c^2 \lambda_k)$", ...
                   "$1/(\sqrt{2\pi} |u|) \exp(-1/2)$", ...
               'Interpreter', 'latex', 'FontSize', 15, 'Location', 'south')
           xline(thresholds(1));xline(thresholds(2))
    figure;
        conditional_pdf = conditional_pdf/max(conditional_pdf);
        semilogx( lamb, conditional_pdf, 'LineWidth', 2 );
        hold on;
        l{1}="$p(\lambda | u, \tau,y)$";
        term2 = term2/max(term2);
        semilogx( lamb, term2, 'LineWidth', 2 ); 
        l{2}="$f_{\alpha/2}(\lambda;1,\cos(\pi\alpha/4)^{2/\alpha},0)$";
        term1 = term1/max(term1);
        semilogx( lamb, term1, 'LineWidth', 2 );
        l{3}="$\mathcal{N}(u_k; 0, 2 c^2 \lambda_k)$";
        xline(thresholds(1),'-.'); xline(thresholds(2),'--')
        l{4}=strcat("$p=0.1$ \% of prior (", sprintf("%.2f", thresholds(1)), ')'); 
        l{5}=strcat("$p=99$ \% of prior (", sprintf("%.2f", thresholds(2)), ')');
        legend(l,'Interpreter', 'latex', 'FontSize', 15)
        
end

