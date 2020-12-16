function [ADMM] = run_ADMM(C_fidelity, Cn, Hn, solver, params, x0)
%-------------------------------------------------------------------------------------
% This function takes the cost functions used in the optimization problem and runs ADMM.
%
% Inputs: C_fidelity: Data fidelity term 
%         C_n: Regularization cost (with implemented prox operator)
%         H_n: Regularization transform domain / matrix
%         solver: function handle for solving the quadratic step in ADMM
%         params:  - name: 'admm'
%                  - maxiter: maximum number of iterations
%                  - relative_tol: to check for convergence
%                  - verbose: 0 or 1
%                  - ItUpOut: cost values are stored every ItUpOut iterations
%                  - iterVerb: display cost after iterVerb iterations  
%                  - rho: Lagrangian penalty parameter
%         x0: initialization
%
% Output: ADMM object
%-------------------------------------------------------------------------------------

ADMM = OptiADMM(C_fidelity, Cn, Hn, params.rho, solver);
ADMM.OutOp = OutputOptiSNR(true, [], params.iterVerb);
ADMM.rho_n = params.rho;
CvOp = TestCvgCostRelative(params.relative_tol);
ADMM.CvOp = CvOp;
ADMM.maxiter = params.maxiter;
ADMM.ItUpOut = params.ItUpOut;
ADMM.verbose = params.verbose;
ADMM.run(x0);

end

