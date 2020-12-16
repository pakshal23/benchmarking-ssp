function [Y] = generate_alpha_stable_rv(params, K)
%---------------------------------------------------------------
% This function generates random variables from an alpha-stable
% distribution with parameters (\alpha, \beta, c, \mu).

% Input: params - (\alpha, \beta, c, \mu)
%        K - number of random variables to be generated
%---------------------------------------------------------------

U = unifrnd(-pi/2, pi/2, [K, 1]);
W = exprnd(1, [K, 1]);

if (params(1) == 1)
   
    e = pi/2;
    X = (1/e).*((pi/2 + params(2).*U).*tan(U) - params(2).*log(((pi/2).*W.*cos(U))./(pi/2 + params(2).*U)));
    Y = params(3).*X + (2/pi)*params(2)*params(3)*log(params(3)) + params(4);
    
else
    
    tau = -params(2)*tan((pi*params(1))/2);
    e = (1/params(1))*atan(-tau);
    X = ((1 + tau^2)^(1/(2*params(1)))).*(sin(params(1).*(U + e))./(cos(U).^(1/params(1)))).*(((cos(U - params(1).*(U + e)))./W).^((1-params(1))/params(1)));
    Y = params(3).*X + params(4);
    
end


end

