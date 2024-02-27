function [x_l2, x_l1, ux_l1] = bayesian_bernoulli_laplace(y, H, L, u0, sig_params, algo_params)
%-----------------------------------------------------------------------------------------
% This function computes the Bayesian estimates (l2-loss, l1-loss) for bernoulli-laplace signals
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
b = sig_params.dist_param(2);
noise_prec = 1/sig_params.noise_var;
lam = 1 - sig_params.dist_param(1);

all_increments = zeros([N,1]);
all_samples = zeros(N, 1);

% Gibbs sampling loop
kk=1;
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
    
 
    if (iterGibbs >= algo_params.burn_in + 1)
        all_increments(:,kk) = increments;
        all_samples(:,kk) = L\increments;
        kk = kk + 1;
    end
end

u_l1 = median(all_increments, 2);
u_quantiles = quantile(all_increments, [0.1, 0.3, 0.5, 0.7, 0.9], 2);
ux_l1 = L\u_quantiles;
x_l1 = quantile(all_samples, [0.1, 0.3, 0.5, 0.7, 0.9], 2);

u_l2 = mean(all_increments, 2);
x_l2 = L\u_l2;

end