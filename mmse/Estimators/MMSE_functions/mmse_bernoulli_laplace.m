function [x_est] = mmse_bernoulli_laplace(y, H, L, u0, sig_params, algo_params)
%-----------------------------------------------------------------------------------------
% This function computes the MMSE estimate for bernoulli-laplace signals
%
% Inputs: y: Data vector
%         H: Forward model
%         L: Regularization matrix
%         u0: initialization
%         sig_params:  - type: 'bernoulli-laplace'
%                      - dist_param: MassProb for the Bernoulli part and b for the Laplace part. [MassProb, b]                                                   
%                      - noise_var: variance of the noise
%         algo_params: - num_iter
%                      - burn_in
%
% Outputs: x_est: MMSE estimate
%--------------------------------------------------------------------------------------

% Computing the effective forward model (since we recover the innovation u)
H_eff = H/L;
[M, N] = size(H_eff);

% Initializations
increments = u0;
q_aux = binornd(ones([N, 1]), 0.5);
aux = zeros([N,1]);
x_est = zeros([N,1]);
b = sig_params.dist_param(2);
noise_prec = 1/sig_params.noise_var;
lam = 1 - sig_params.dist_param(1);

% Gibbs sampling loop
for iterGibbs = 2:algo_params.num_iter
    ind0 = (q_aux==0);
    aux(ind0) = exprnd(2*b*b, [sum(ind0), 1]);
    ind1 = (q_aux==1);
    aux(ind1) = sample_gig(b, increments(ind1));
    
    for i = 1:N
        
        q = sample_location(i, y, H_eff, q_aux, aux, sig_params.noise_var, lam);
        q_aux(i) = q;
        
    end
    
    rem_ind = (q_aux==1);
    increments(q_aux==0) = 0;
    H_trunc = H_eff(:, rem_ind);
    new_len = size(H_trunc, 2);
    InvCov = noise_prec * (H_trunc')*H_trunc + diag(1./aux(rem_ind));
    FacChol = chol(InvCov);
    increments(rem_ind) = FacChol \ ( (FacChol')\(noise_prec * (H_trunc')*y) + randn(new_len,1) );
    
    sig = L\increments;
 
    if (iterGibbs >= algo_params.burn_in + 1)
        x_est = ( (iterGibbs - algo_params.burn_in - 1) * x_est + sig ) / (iterGibbs - algo_params.burn_in);
    end
end

end