function [x_est, exp_mse] = mmse_student(y, H, L, u0, sig_params, algo_params)

% Computing the effective forward model (since we recover the innovation u)
%L_inv = inv(L);
%H_eff = H*L_inv;
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

exp_mse = 0;

% Gibbs sampling loop
for iterGibbs = 2:algo_params.num_iter

    ParamB = 0.5 * ( beta + increments.^2 );
    VarAux = gamrnd( alpha(ones([N,1])) , 1./ParamB);

    InvCov = noise_prec * HTH + diag(VarAux);
    FacChol = chol(InvCov);
    increments = FacChol \ ( (FacChol')\(noise_prec * HTy) + randn([N,1]) );

    %sig = L_inv*increments;
    sig = L\increments;
    
    % Computing mean of the samples from the posterior distribution
    if (iterGibbs >= algo_params.burn_in + 1)
       x_est = ( (iterGibbs - algo_params.burn_in - 1) * x_est + sig ) / (iterGibbs - algo_params.burn_in);
    end

end

for iter = 1:5000
       
    ParamB = 0.5 * ( beta + increments.^2 );
    VarAux = gamrnd( alpha(ones([N,1])) , 1./ParamB);

    InvCov = noise_prec * HTH + diag(VarAux);
    FacChol = chol(InvCov);
    increments = FacChol \ ( (FacChol')\(noise_prec * HTy) + randn([N,1]) );

    %sig = L_inv*increments;
    sig = L\increments;

    exp_mse = exp_mse + norm(sig-x_est,2)^2;

end

exp_mse = exp_mse/5000;

end