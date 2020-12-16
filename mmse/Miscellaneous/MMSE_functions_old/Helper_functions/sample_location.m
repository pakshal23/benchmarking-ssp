function [q] = sample_location(i, y, H, q_aux, aux, noise_var, lam)

[M, N] = size(H);

% q0 = q_aux; q0(i) = 0;
% ind0 = (q0==1);
% H_bar = H(:, ind0);
% B0 = noise_var.*eye(M) + H_bar*diag(aux(ind0))*(H_bar');
% B0_inv = inv(B0);
% 
q1 = q_aux; q1(i) = 1;
ind1 = (q1==1);
H_bar = H(:, ind1);
B1 = noise_var.*eye(M) + H_bar*diag(aux(ind1))*(H_bar');
%B1_inv = inv(B1);
% 
% del_0 = y'*(B1_inv-B0_inv)*y + log(det(B1)/det(B0)) + 2*log((1-lam)/lam);

H_i = H(:, i);
T1 = (-1/aux(i)) + H_i'*(B1\H_i);
%T1 = (-1/aux(i)) + H_i'*(B1_inv*H_i);
gamma_1 = H_i'*(B1\y);
%gamma_1 = H_i'*(B1_inv*y);

del_1 = (-1/T1)*(gamma_1'*gamma_1) + log(-1*aux(i)*T1) - 2*log((1-lam)/lam);

%p = 1/(1 + exp(-del_0/2));
p = 1/(1 + exp(-del_1/2));

%q = binornd(1, 1-p);
q = binornd(1, p);


end

