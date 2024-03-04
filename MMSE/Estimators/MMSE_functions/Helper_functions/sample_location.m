function [q] = sample_location(i, y, H, q_aux, aux, noise_var, lam)

[M, N] = size(H);

q1 = q_aux; q1(i) = 1;
ind1 = (q1==1);
H_bar = H(:, ind1);
B1 = noise_var.*eye(M) + H_bar*diag(aux(ind1))*(H_bar');

H_i = H(:, i);
T1 = (-1/aux(i)) + H_i'*(B1\H_i);
gamma_1 = H_i'*(B1\y);
del_1 = (-1/T1)*(gamma_1'*gamma_1) + log(-1*aux(i)*T1) - 2*log((1-lam)/lam);

p = 1/(1 + exp(-del_1/2));
q = binornd(1, p);

end

