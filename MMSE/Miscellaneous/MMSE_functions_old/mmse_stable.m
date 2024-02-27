function [x_est] = mmse_stable(y, H, L, u0, sig_params, algo_params)

% Computing the effective forward model (since we recover the innovation u)
%L_inv = inv(L);
%H_eff = H*L_inv;
H_eff = H/L;
[M, N] = size(H_eff);


% Initializations
increments = u0;
x_est = zeros([N,1]);
VarAux2 = zeros([N,1]);

params = sig_params.dist_param;
noise_prec = 1/sig_params.noise_var;


% Precomputations
HTH = H_eff'*H_eff;
HTy = H_eff'*y;
HHT = H_eff*H_eff';


if (params(1) == 2)
   
    u_est = (2*(params(3)^2)).*H_eff'*(inv((2*(params(3)^2)).*HHT + sig_params.noise_var.*eye(M)))*y;
    x_est = L\u_est;
    
else
    
    if (strcmp(algo_params.mode, 'slice'))
        
        % Gibbs sampling loop
        for iterGibbs = 2:algo_params.num_iter
            
            disp(num2str(iterGibbs));
            
            [VarAux, VarAux2] = sample_aux_variable_stable_slice(increments, VarAux2, params);

            InvCov = noise_prec * HTH + diag(1./(2.*(params(3)^2).*VarAux));
            FacChol = chol(InvCov);
            increments = FacChol \ ( (FacChol')\(noise_prec * HTy) + randn([N,1]) );

            %sig = L_inv*increments;
            sig = L\increments;

            % Computing mean of the samples from the posterior distribution
            if (iterGibbs >= algo_params.burn_in + 1)
               x_est = ( (iterGibbs - algo_params.burn_in - 1) * x_est + sig ) / (iterGibbs - algo_params.burn_in);
            end

        end
        
    else
        
        %grid = linspace(1e-3,10,1024);
        
        % Gibbs sampling loop
        for iterGibbs = 2:algo_params.num_iter

            %disp(num2str(iterGibbs));

            %VarAux = sample_aux_variable_stable(increments, params);
            VarAux = sample_aux_variable_stable_grid(increments, params);

            InvCov = noise_prec * HTH + diag(1./(2.*(params(3)^2).*VarAux));
            FacChol = chol(InvCov);
            increments = FacChol \ ( (FacChol')\(noise_prec * HTy) + randn([N,1]) );

            %sig = L_inv*increments;
            sig = L\increments;

            % Computing mean of the samples from the posterior distribution
            if (iterGibbs >= algo_params.burn_in + 1)
               x_est = ( (iterGibbs - algo_params.burn_in - 1) * x_est + sig ) / (iterGibbs - algo_params.burn_in);
            end

        end
        
    end
    
    
    
    
    
end




end