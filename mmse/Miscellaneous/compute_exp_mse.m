function [exp_mmse, exp_l2, exp_log] = compute_exp_mse(y, H, L, u0, sig_params, algo_params, x_mmse, x_l2, x_log)

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

exp_mmse = 0;
exp_l2 = 0;
exp_log = 0;

% Gibbs sampling loop
for iterGibbs = 2:algo_params.num_iter

    ParamB = 0.5 * ( beta + increments.^2 );
    VarAux = gamrnd( alpha(ones([N,1])) , 1./ParamB);

    InvCov = noise_prec * HTH + diag(VarAux);
    FacChol = chol(InvCov);
    increments = FacChol \ ( (FacChol')\(noise_prec * HTy) + randn([N,1]) );

end

for iter = 1:10000
       
    ParamB = 0.5 * ( beta + increments.^2 );
    VarAux = gamrnd( alpha(ones([N,1])) , 1./ParamB);

    InvCov = noise_prec * HTH + diag(VarAux);
    FacChol = chol(InvCov);
    increments = FacChol \ ( (FacChol')\(noise_prec * HTy) + randn([N,1]) );

    %sig = L_inv*increments;
    sig = L\increments;

    exp_mmse = exp_mmse + norm(sig - x_mmse,2)^2;
    exp_l2 = exp_l2 + norm(sig - x_l2,2)^2;
    exp_log = exp_log + norm(sig - x_log,2)^2;

end

exp_mmse = exp_mmse/10000;
exp_l2 = exp_l2/10000;
exp_log = exp_log/10000;

end