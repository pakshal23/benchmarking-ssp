function [x_est] = mmse_stable(y, H, L, u0, sig_params, algo_params)
%-----------------------------------------------------------------------------------------
% This function computes the MMSE estimate for bernoulli-laplace signals
%
% Inputs: y: Data vector
%         H: Forward model matrix
%         L: Regularization matrix
%         u0: initialization (if empty, will default to Gaussian prior)
%         sig_params: Signal parameters object, containing
%                      - type: 'alpha-stable'
%                      - dist_param: [alpha, beta, gamma, delta]                
%                      - noise_var: variance of the noise
%         algo_params: Algorithm parameters object, containing
%                      - num_iter
%                      - burn_in
%
% Outputs: x_est: MMSE estimate
%--------------------------------------------------------------------------------------

    % Extract relevant parameters
    params = sig_params.dist_param;
    noise_prec = 1/sig_params.noise_var;

    % Computing the effective forward model (since we will sample the 
    % innovation u)
    H_eff = H/L;
    [M, N] = size(H_eff);

    % Precomputations
    HTH = H_eff'*H_eff;
    HTy = H_eff'*y;
    HHT = H_eff*H_eff';

    % Initializations (default to the Gaussian case)
    if isempty(u0) || params(1) == 2
        % Closed form solution for the Gaussian case
        increments = (2*(params(3)^2)) .* H_eff' * (inv((2*(params(3)^2)) .* HHT + ...
                 sig_params.noise_var.*eye(M))) * y;
        x_est = L\increments;
    else
        increments = u0;
        x_est = zeros([N,1]);
    end
    
    
    if (params(1) ~= 2)      
        % Gibbs sampling loop
        for iterGibbs = 2:algo_params.num_iter
            % Sample auxiliary variable
            VarAux = sample_aux_variable_stable(increments, params);
            % Sample high-dimensional Gaussian using (Rue 2001)
            InvCov = noise_prec * HTH + diag(1./(2.*(params(3)^2).*VarAux));
            FacChol = chol(InvCov);
            increments = FacChol \ ( (FacChol')\(noise_prec * HTy) + randn([N,1]) );
            % Compute signal for the sampled increments
            sig = L\increments;
            % Computing mean of the samples from the posterior distribution
            if (iterGibbs >= algo_params.burn_in + 1)
               x_est = ( (iterGibbs - algo_params.burn_in - 1) * x_est + sig ) / (iterGibbs - algo_params.burn_in);
            end 
        end
    end
end