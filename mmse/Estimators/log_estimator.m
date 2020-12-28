function [x_est, opt_cost, num_iter] = log_estimator(y, H, L, epsilon, lambda, params, x0)
%-----------------------------------------------------------------------------------------------
% Implementation of the log-regularized estimator
%
% Inputs: y: Data vector
%         H: Forward model
%         L: Regularization matrix
%         epsilon: For the log penalty
%         lambda: Regularization parameter
%         params:  - name: 'fista' or 'admm'
%                  - maxiter: maximum number of iterations
%                  - relative_tol: to check for convergence
%                  - verbose: 0 or 1
%                  - ItUpOut: cost values are stored every ItUpOut iterations
%                  - iterVerb: display cost after iterVerb iterations  
%                  - rho: for admm
%                  - gam: for fista 
%         x0: initialization
%
% Outputs: x_est: solution
%          opt_cost: optimal cost
%          num_iter: number of iterations till convergence
%-----------------------------------------------------------------------------------------------

% ADMM
if(strcmp(params.name, 'admm'))
    % Create the look-up table for the prox operator
    tab = generateLUT(lambda/params.rho, epsilon);
    tfun = @(x) (interp1q(tab(:, 1),tab(:, 2),abs(x)).*sign(x));
    
    % ADMM
    H_op = LinOpMatrix(H);
    L_op = LinOpMatrix(L);
    C_fidelity = 2*CostL2(H_op.sizeout, y) * H_op;
    C_fidelity.doPrecomputation=1;
    C_regul = lambda*CostLog(L_op.sizeout, epsilon, tfun);
    Cn = {C_regul};
    Hn = {L_op};
    solver = @(z_n, rho_n, x) (2*(H'*H) + rho_n*(L'*L)) \ (2*H'*y + rho_n*L'*z_n{1});
    ADMM = run_ADMM(C_fidelity, Cn, Hn, solver, params, x0);
    x_est = ADMM.xopt;
    opt_cost = ADMM.cost.apply(x_est);
    num_iter = ADMM.niter;

% FISTA    
elseif (strcmp(params.name, 'fista'))
    
    % FISTA
    K = size(L, 1);
    %L_inv = inv(L);
    %H_op = LinOpMatrix(H*L_inv);
    H_op = LinOpMatrix(H/L);
    C_fidelity = 2*CostL2(H_op.sizeout, y) * H_op;
    C_fidelity.doPrecomputation=1;

    % Create the look-up table for the prox operator
    tab = generateLUT(lambda*(1/C_fidelity.lip), epsilon);
    tfun = @(x) (interp1q(tab(:, 1),tab(:, 2),abs(x)).*sign(x));
    
    C_regul = lambda*CostLog([K, 1], epsilon, tfun);
    FBS = run_FBS(C_fidelity, C_regul, params, x0);
    u_est = FBS.xopt;
    opt_cost = FBS.cost.apply(u_est);
    num_iter = FBS.niter;
    %x_est = L_inv*u_est;
    x_est = L\u_est;
    
end

end