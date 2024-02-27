function [out] = sample_aux_variable_stable(increments, params, approx )
    if nargin < 3
        approx = True;
    end

    % Check for zero increments
    if any(increments==0)
        error( "Some u is 0, this will never end" )
    end

    % Extract sizes and preallocate output
    N = length(increments);
    out = zeros([N,1]);

    % Compute parameters for candidate distribution
    mix_params = [params(1)/2, 1, (cos((pi*params(1))/4)^(2/params(1))), 0];

    % Compute upper limit
    C = (1./abs(increments))*(1/sqrt(2*pi))*exp(-0.5);

    % If approximation mode is on, compute thresholds on increments to switch
    % to approximated operation
    if approx
        thresholds = stblinv([0.01,0.99], mix_params(1), mix_params(2), mix_params(3), mix_params(4)); 
    end

    for i = 1:N
        % Implementing tail approximations from Godsill 1999
        % Data is much larger than the prior, use assymptotic result for
        % alpha-stable distributions, end up sampling from IG
        if approx && (increments(i) >= thresholds(2))
            % Sample from inverse Gamma distribution IG((alpha+1)/2, u^2/2)
            % Pending checks! MATLAB parametrizations!
            gamma_sample = gamrnd((mix_params(1)+1)/2, 2/increments(i)^2);
            out(i) = 1/gamma_sample;
        % Data is much smaller than the prior, ignore values to the left of
        % the prior and re-scale likelihood to get better acceptance
        % probabilities
        elseif approx && (increments(i) <= thresholds(1))
            % Compute new upper limit (likelihood value at lower threshold)
            ul = (1 / sqrt(4*pi*(params(3)^2)*thresholds(1) )) * exp( -(increments(i)^2) / (4*(params(3)^2)*thresholds(1)) );
            % Sample constrained to above lower threshold
            while(1)
                candidate_sample = generate_alpha_stable_rv_larger(mix_params, 1);
                uniform_sample = unifrnd(0, ul);
                prob_val = (1/sqrt(4*pi*(params(3)^2)*candidate_sample))*exp(-(increments(i)^2)/(4*(params(3)^2)*candidate_sample));
                if (uniform_sample < prob_val)
                    out(i) = candidate_sample;
                    break;
                end
            end
        % Good case, rejection should not take too long
        else
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
end

    function val = generate_alpha_stable_rv_larger(mix_params, value)
        val = value-0.1;
        while(val<value)
            val = generate_alpha_stable_rv(mix_params, 1);
        end
    end