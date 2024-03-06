function [FBS] = run_FBS(C_fidelity, C_regul, params, x0)
%--------------------------------------------------------------------------------------------
% This function takes the cost functions used in the optimization problem and runs FISTA.
%
% Inputs: C_fidelity: Data fidelity term 
%         C_regul: Regularization cost (with implemented prox operator)
%         params:  - name: 'fista'
%                  - maxiter: maximum number of iterations
%                  - relative_tol: to check for convergence
%                  - verbose: 0 or 1
%                  - ItUpOut: cost values are stored every ItUpOut iterations
%                  - iterVerb: display cost after iterVerb iterations  
%                  - gam: step-size
%         x0: initialization
%
% Output: ADMM object
%--------------------------------------------------------------------------------------------
%%

FBS = OptiFBS(C_fidelity, C_regul);
FBS.fista = true;
%FBS.OutOp = OutputOptiSNR(true, [], params.iterVerb);
%FBS.OutOp = OutputOpti(true, [], params.iterVerb);
CvOp = TestCvgCostRelative(params.relative_tol);
FBS.CvOp = CvOp;
FBS.maxiter = params.maxiter;
FBS.ItUpOut = params.ItUpOut;
FBS.verbose = params.verbose;
%FBS.gam  % We let the library set 'gam' on its own
FBS.run(x0);

end

