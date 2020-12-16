function [out, out2] = sample_aux_variable_stable_slice(increments, var_aux2, params)

N = length(increments);
out = zeros([N,1]);
fin_prob_val = zeros([N,1]);

mix_params = [params(1)/2, 1, (cos((pi*params(1))/4)^(2/params(1))), 0];

for i = 1:N
   
    while(1)
        candidate_sample = generate_alpha_stable_rv(mix_params, 1);
        %uniform_sample = unifrnd(0, C(i));
        prob_val = (1/sqrt(4*pi*(params(3)^2)*candidate_sample))*exp(-(increments(i)^2)/(4*(params(3)^2)*candidate_sample));
        if (prob_val > var_aux2(i))
            out(i) = candidate_sample;
            fin_prob_val(i) = prob_val;
            break;
        end
    end
    
end

out2 = unifrnd(zeros([N,1]), fin_prob_val);



end