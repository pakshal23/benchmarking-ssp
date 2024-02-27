function [x_est] = mmse_student(y, H, L, u0, sig_params, algo_params)
%-----------------------------------------------------------------------------------------
% This function computes the MMSE estimate for student's t signals
%
% Inputs: y: Data vector
%         H: Forward model
%         L: Regularization matrix
%         u0: initialization
%         sig_params:  - type: 'student'
%                      - dist_param: alpha (alpha=1 corresponds to the Cauchy distribution) [ \propto (1/(1 + x^2))^(alpha)]                                                   
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
x_est = zeros([N,1]);
beta = 1;
alpha = sig_params.dist_param;
noise_prec = 1/sig_params.noise_var;

% Precomputations
HTH = H_eff'*H_eff;
HTy = H_eff'*y;

% Gibbs sampling loop
for iterGibbs = 2:algo_params.num_iter
    ParamB = 0.5 * ( beta + increments.^2 );
    VarAux = gamrnd( alpha(ones([N,1])) , 1./ParamB);

    InvCov = noise_prec * HTH + diag(VarAux);
    FacChol = chol(InvCov);
    increments = FacChol \ ( (FacChol')\(noise_prec * HTy) + randn([N,1]) );

    sig = L\increments;
    
    % Computing mean of the samples from the posterior distribution
    if (iterGibbs >= algo_params.burn_in + 1)
       x_est = ( (iterGibbs - algo_params.burn_in - 1) * x_est + sig ) / (iterGibbs - algo_params.burn_in);
    end
end

end