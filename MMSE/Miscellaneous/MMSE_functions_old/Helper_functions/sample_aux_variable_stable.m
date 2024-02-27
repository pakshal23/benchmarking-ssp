function [out] = sample_aux_variable_stable(increments, params)

N = length(increments);
out = zeros([N,1]);

mix_params = [params(1)/2, 1, (cos((pi*params(1))/4)^(2/params(1))), 0];
C = (1./abs(increments)).*(1/sqrt(2*pi)).*exp(-0.5);

for i = 1:N
   
    while(1)
        candidate_sample = generate_alpha_stable_rv(mix_params, 1);
        uniform_sample = unifrnd(0, C(i));
        prob_val = (1/sqrt(4*pi*(params(3)^2)*candidate_sample))*exp(-(increments(i)^2)/(4*(params(3)^2)*candidate_sample));
        if (uniform_sample < prob_val)
            out(i) = candidate_sample;
            break;
        end
    end
    
end

end

