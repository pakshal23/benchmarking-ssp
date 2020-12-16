function [x_est, opt_cost, num_iter] = l1_estimator(y, H, L, lambda, params, x0)
%-------------------------------------------------------------------------------------------
% Implementation of the l1-regularized estimator
%
% Inputs: y: Data vector
%         H: Forward model
%         L: Regularization matrix
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
%-------------------------------------------------------------------------------------------

% ADMM
if(strcmp(params.name, 'admm'))
    H_op = LinOpMatrix(H);
    L_op = LinOpMatrix(L);
    C_fidelity = 2*CostL2(H_op.sizeout, y) * H_op;
    C_fidelity.doPrecomputation=1;
    C_regul = lambda*CostL1(L_op.sizeout);
    Cn = {C_regul};
    Hn = {L_op};
    solver = @(z_n, rho_n, x) (2*(H'*H) + rho_n*(L'*L)) \ (2*H'*y + rho_n*L'*z_n{1});
    ADMM = run_ADMM(C_fidelity, Cn, Hn, solver, params, x0);
    x_est = ADMM.xopt;
    opt_cost = ADMM.cost.apply(x_est);
    num_iter = ADMM.niter;
    
% FISTA    
elseif (strcmp(params.name, 'fista'))
    K = size(L, 1);
    %L_inv = inv(L);
    %H_op = LinOpMatrix(H*L_inv);
    H_op = LinOpMatrix(H/L);
    C_fidelity = 2*CostL2(H_op.sizeout, y) * H_op;
    C_fidelity.doPrecomputation=1;
    C_regul = lambda*CostL1([K, 1]);
    FBS = run_FBS(C_fidelity, C_regul, params, x0);
    u_est = FBS.xopt;
    opt_cost = FBS.cost.apply(u_est);
    num_iter = FBS.niter;
    %x_est = L_inv*u_est;
    x_est = L\u_est;
    
end

end

