function [s] = generate_discrete_levy_process(handles)
%--------------------------------------------------------------------------------------------------------------------------------
% This function generates a discrete signal s such that Ds = u. The matrix
% D is the finite-difference operator (with zero-boundary conditions). The vector u
% follows one of the following distributions (required parameters of the distibution are also indicated):
%
% 1) Gaussian (zero mean): variance (var)   [exp(-x^2 / (2*var^2)) / (2*pi*var^2)^0.5]
% 2) Laplace: b   [exp(-|x| / b) / (2*b)]
% 3) Student: alpha (alpha=1 corresponds to the Cauchy distribution) [ \propto (1/(1 + x^2))^(alpha)]
% 4) Bernoulli-Laplace: MassProb for the Bernoulli part and b for the Laplace part. [MassProb, b]
% 5) Alpha-Stable: [alpha, beta, gamma, delta]
%
% Input: handles
% handles.Prior: 'gaussian', 'laplace', 'student', 'bernoulli-laplace', 'alpha-stable'
% handles.K: Length of the signal;
% handles.Dist_Param: Parameters for the chosen distribution of u; For 'bernoulli-laplace': handles.Dist_Param = [mass_prob, b]
%
% Output: signal s [K x 1]
%---------------------------------------------------------------------------------------------------------------------------------

switch handles.Prior
    
    case 'gaussian'
        
        u = sqrt(handles.Dist_Param).*randn([handles.K, 1]);
        
    case 'laplace'
        
        aux = rand([handles.K,1])-0.5;
        u = -handles.Dist_Param .* sign(aux) .* (log(1 - 2.*abs(aux)));     
           
    case 'student'

        aux = gamrnd(handles.Dist_Param - 0.5, 2, [handles.K, 1]);
        u = randn([handles.K, 1]) ./ sqrt(aux);
              
    case 'bernoulli-laplace'
        
        aux = rand([handles.K,1]) < (1 - handles.Dist_Param(1));
        u = zeros([handles.K,1]);
        
        aux_lap = rand([length(find(aux==1)),1])-0.5;
        u(aux==1) = -handles.Dist_Param(2) .* sign(aux_lap) .* (log(1 - 2.*abs(aux_lap)));
        
    case 'alpha-stable'
        
        %u = generate_alpha_stable_rv(handles.Dist_Param, handles.K);
        u = stblrnd(handles.Dist_Param(1), handles.Dist_Param(2), handles.Dist_Param(3), handles.Dist_Param(4), [handles.K, 1]);
        
end

s = cumsum(u);    

end

