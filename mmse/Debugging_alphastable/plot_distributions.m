function [] = plot_distributions(alpha, u, maxlam)
if nargin < 3
    maxlam = 5;
    if nargin < 2
        u = 0.5;
        if nargin < 1
            alpha = 1.5;
        end
    end
end

    % \lambda axis
    lamb = (0.001:0.001:maxlam);
    
    % Distribution we want to sample from 
    % Term arising from Gaussian mixture
    term1 = (1 ./ sqrt(4*pi*lamb) ) .* exp( -(u^2) ./ (4*lamb) );
    % \alpha^'-stable term
    stable_pdf = makedist('Stable','alpha',alpha/2,'beta',1,'gam', (cos(pi*alpha/4))^(2/alpha),'delta',0);
    term2 = pdf(stable_pdf, lamb);
    % Distribution
    conditional_pdf = term1.*term2;

    % Constant C for rejection sampling? 
    C = (1/sqrt(2*pi)) * (1/abs(u)) * exp(-0.5);

    figure;
        subplot(1, 2, 1)
            plot( lamb, conditional_pdf, 'LineWidth', 2 );
            title("$p(\lambda | u, \tau,y)$", ...
                    'Interpreter', 'latex', 'FontSize', 15)
        subplot(2, 2, 2)
            plot( lamb, term2, 'LineWidth', 2 ); % Could multiply by C for comparable height
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
end

