function [samples] = sample_gig(sigma, b_values)

K = length(b_values);
samples = zeros([K, 1]);

for i = 1:K
    samples(i,1) = gigrnd(0.5, 1/(sigma*sigma), (b_values(i))^2, 1);
end

end

