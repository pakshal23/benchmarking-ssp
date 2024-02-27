function [t, N] = generateLUT(lambda, epsilon, xmax, delta, xinf)
% GENERATELUT generates lookup table for log prior.
%
% t = generateLUT(lambda, epsilon, xmax, delta, xinf)
%
% This function generates lookup table for the functional:
% f(x) = 0.5*(x-y)^2 + lambda*log(x^2 + epsilon)
%
% Input:
% - lambda: regularization parameter (0 <= lambda < Inf)
% - epsilon: sparsity parameter (0 <= epsilon < Inf)
% - xmax: maximal output value (default: xmax = 100)
% - delta: output sampling step abs(x_(i+1) - x_i) (default: 0.01)
% - xinf: represents infinity (default: 1e6)
%
% Output:
% - t: lookup table t = [input, output], of dimension [N x 2]
% - N: number of elements in the table
%
% Ulugbek Kamilov, Emrah Bostan, BIG @ EPFL, 2011.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set default parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(~exist('delta', 'var'))
    delta = 0.01;
end

if(~exist('xmax', 'var'))
    xmax = 100;
end

if(~exist('xinf', 'var'))
    xinf = 1e6;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Generate initial output points
x = [(0:delta:xmax)'; xinf];

% Check if one-to-one
if(lambda <= 4*epsilon)
    % If one-to-one generate the table directly
    y = criticalPoints(x, lambda, epsilon);
else
    % Generate critical points
    y = criticalPoints(x, lambda, epsilon);
    
    % Table 1
    a1 = sqrt(lambda - epsilon - sqrt(lambda*(lambda - 4*epsilon)));
    x1 = x(x <= a1);
    y1 = y(x <= a1);
    
    % Table 2
    a2 = sqrt(lambda - epsilon + sqrt(lambda*(lambda - 4*epsilon)));
    x2 = x(x >= a2);
    y2 = y(x >= a2);
    
    % Intialize combined tables
    y = sort([y1; y2]);
    x = zeros(size(y));
    
    % Indices of regions of the table
    i1 = y < y2(1);
    i2 = y > y1(end);
    i3 = y >= y2(1) & y <= y1(end);
    
    % Set first and third regions
    x(i1) = x1(y1 < y2(1));
    x(i2) = x2(y2 > y1(end));
    
    % Set the second region
    yy = y(i3);
    xx1 = interp1q(y1, x1, yy);
    xx2 = interp1q(y2, x2, yy);
    
    f1 = functional(yy, xx1, lambda, epsilon);
    f2 = functional(yy, xx2, lambda, epsilon);
    
    x(i3) = ((f1 < f2) .* xx1) + ((f1 >= f2) .* xx2);
end

% Combine input and output into single table
t = [y, x];
N = size(t, 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function f = functional(y, x, lambda, epsilon)
% Functional we are trying to minimize
f = 0.5*((x-y).^2) + lambda*log((x.^2)+epsilon);

function y = criticalPoints(x, lambda, epsilon)
% Returns critical points of the functional
y = x .* (1 + 2*lambda ./ ((x.^2)+epsilon));