function x_out = TMP_MMSE_Levy(y , Samp_Grid , prior , noise_Var , varargin)

% x_out = MPT_MMSE_Levy(y , Samp_Grid , prior , noise_Var , varargin)
%
% The function implements the MMSE denoiser for a Levy process. The
% statistics of the innovation process are given in "prior". The
% implementation of the MMSE is based on the message passing algorithm were
% the messages are the pdf curves (samples). The term 'TMP' in the name of 
% the function is the abbreviation of 'Time-domain Message Passing' (in 
% contrast to 'Frequency-domain Message Passing' where the messages are the 
% curves or samples of the characteristic functions.)
%
% "y" is the vector of noisy measurements.
%
% "Samp_Grid" is a vector that contains the points at which we sample the
% probability density functions.
%
% "prior" is a structure which includes the statistics of the innovation
% process:
%       1) 'prior.type' show the pdf type of the finite differences and
%          accepts the following options:
%               'g'  or 'gaussian'              => zero-mean Gaussian pdf        
%               'l'  or 'laplace'               => Laplace pdf
%               's'  or 'student'               => Student's-t pdf
%               'a'  or 'alpha_stable'          => Symmetric alpha-stable pdf 
%                                                  with shape factor 1
%               'pg' or 'poisson-gaussian'      => Impulsive Poisson pdf with 
%                                                  zero-mean Gaussian amplitudes
%               'pl' or 'poisson-laplace'       => Impulsive Poisson pdf with 
%                                                  Laplace amplitudes
%               'ps' or 'poisson-student'       => Impulsive Poisson pdf with 
%                                                  Student's-t amplitudes
%               'pa' or 'poisson-alpha_stable'  => Impulsive Poisson pdf with 
%                                                  Symmetric alpha-stable  
%                                                  with shape factor 1
%
%       2) 'prior.param' which is the only degree of freedom in the given
%          pdf. In more details, it is the varinace for Gaussian pdf's ('g'
%          or 'pg'), $b$ parameter in Laplace distribution (for 'l' or
%          'pl'), $q$ parameter in Student's-t distribution (for 's' or
%          'ps') and $alpha$ in alpha-stable distributions ('a' or 'pa').
%
%       3) 'prior.rho' is the probability of having a non-zero sample in
%          the Poisson-type innovations. For non-Poisson innovations this
%          parameter can be ignored.
%
% "noise_Var" is the variance of the additive white Gaussian noise of the
% observed samples. 
%
% Other parameter listed below are optional in the sense that if they are
% not given, a default value is used:
%
% ADMM_log(... , 'Anim_En' , flag , ...)
% "flag" shows if the results in the intermediate iterations are plotted or
% not. The default value is 'false'. Other inputs enable the plot.
%
% ADMM_log(... , 'Orig_Sig' , x_orig , ...)
% "x_orig" is the original noiseless signal which we would like to obtain
% by this regularization method. It is only required when the animation 
% obtion is enabled.



% updating with the given inputs
[flag , x_orig]     = process_options(varargin , 'Anim_En' , 'false' , 'Orig_Sig' , []);



% converting the inpits to the appropriate format
if size(y , 1) == 1
    ys  = y.';
elseif size(y , 2) == 1
    ys  = y;
else
    error('!!! "y" should be a vector !!!')
    return
end

if size(x_orig , 1) == 1
    x_orig  = x_orig.';
else
    x_orig  = x_orig;
end






% initial parameters
n           = length(ys);                           % Signal length
T           = n - 1;                                % Number of iterations
[cl , cr]   = initialize(prior , Samp_Grid , n);    % Initialize messages
x_cur       = ys;                                   % Initialize estimate
p           = priordist(Samp_Grid , prior);         % Prior probability


%%%%%%%%%%
% main iterations
%%%%%%%%%%%%%%%%%%%%%%
if strcmpi(flag , 'false')  % no animation
    
    for t = 1:T
        % Left side update
        phi         = gaussian( Samp_Grid , ys(t) , noise_Var);
        vl          = normalize( phi .* cl(t , :) );
        cl(t+1, :)  = normalize( convolve(p , vl) );
        
        % Right side update
        phi         = gaussian(Samp_Grid , ys(n+1-t) , noise_Var );
        vr          = normalize( phi .* cr(n+1-t , :) );
        cr(n-t, :)  = normalize( convolve(p , vr) );
        
        % Perform left estimation
        apostl      = normalize(cl(t+1 , :) .* cr(t+1 , :) .* gaussian(Samp_Grid , ys(t+1) , noise_Var));
        x_cur(t+1)  = sum(Samp_Grid .* apostl);
        
        % Perform right estimation
        apostr      = normalize(cl(n-t , :) .* cr(n-t , :) .* gaussian(Samp_Grid , ys(n-t) , noise_Var));
        x_cur(n-t)  = sum(Samp_Grid .* apostr);
        
    end
    
    
else
    
    
    if ~isempty(x_orig)     % with animition and with knowing the oracle solution
        
        for t = 1:T
            % Left side update
            phi         = gaussian( Samp_Grid , ys(t) , noise_Var);
            vl          = normalize( phi .* cl(t , :) );
            cl(t+1, :)  = normalize( convolve(p , vl) );
            
            % Right side update
            phi         = gaussian(Samp_Grid , ys(n+1-t) , noise_Var );
            vr          = normalize( phi .* cr(n+1-t , :) );
            cr(n-t, :)  = normalize( convolve(p , vr) );
            
            % Perform left estimation
            apostl      = normalize(cl(t+1 , :) .* cr(t+1 , :) .* gaussian(Samp_Grid , ys(t+1) , noise_Var));
            x_cur(t+1)  = sum(Samp_Grid .* apostl);
            
            % Perform right estimation
            apostr      = normalize(cl(n-t , :) .* cr(n-t , :) .* gaussian(Samp_Grid , ys(n-t) , noise_Var));
            x_cur(n-t)  = sum(Samp_Grid .* apostr);
            
            
            % Animation
            plot([1:n]' , x_orig , [1:n]' , ys , '.' , [1:n]' , x_cur , '-.', 'LineWidth', 1.5)
            grid on;
            hold on;
            plot([t+1-0.5 t+1-0.5] , [min(ys), max(ys)] , 'g');
            plot([t+1+0.5 t+1+0.5] , [min(ys), max(ys)] , 'g');
            plot([n-t-0.5 n-t-0.5] , [min(ys), max(ys)] , 'c');
            plot([n-t+0.5 n-t+0.5] , [min(ys), max(ys)] , 'c');
            SNR         = 10*log10( mean(x_orig.^2) / mean( (x_orig - x_cur).^2 ));
            title(sprintf('[#iter. = %d/%d      SNR = %f dB]', t , T , SNR));
            legend('Main', 'Noisy', 'Est.');
            axis([-0.5 , length(ys)+0.5 , min(ys)-1 , max(ys)+1])
            hold off;
            getframe;
            
        end
        
    else                    % with animition but without knowing the oracle solution
        
        for t = 1:T
            % Left side update
            phi         = gaussian( Samp_Grid , ys(t) , noise_Var);
            vl          = normalize( phi .* cl(t , :) );
            cl(t+1, :)  = normalize( convolve(p , vl) );
            
            % Right side update
            phi         = gaussian(Samp_Grid , ys(n+1-t) , noise_Var );
            vr          = normalize( phi .* cr(n+1-t , :) );
            cr(n-t, :)  = normalize( convolve(p , vr) );
            
            % Perform left estimation
            apostl      = normalize(cl(t+1 , :) .* cr(t+1 , :) .* gaussian(Samp_Grid , ys(t+1) , noise_Var));
            x_cur(t+1)  = sum(Samp_Grid .* apostl);
            
            % Perform right estimation
            apostr      = normalize(cl(n-t , :) .* cr(n-t , :) .* gaussian(Samp_Grid , ys(n-t) , noise_Var));
            x_cur(n-t)  = sum(Samp_Grid .* apostr);
            
            
            % Animation
            plot([1:n]' , ys , '.' , [1:n]' , x_cur , '-.', 'LineWidth', 1.5)
            grid on;
            hold on;
            plot([t+1-0.5 t+1-0.5] , [min(ys), max(ys)] , 'g');
            plot([t+1+0.5 t+1+0.5] , [min(ys), max(ys)] , 'g');
            plot([n-t-0.5 n-t-0.5] , [min(ys), max(ys)] , 'c');
            plot([n-t+0.5 n-t+0.5] , [min(ys), max(ys)] , 'c');
            title(sprintf('[#iter. = %d/%d]', t , T));
            legend('Noisy', 'Est.');
            axis([-0.5 , length(ys)+0.5 , min(ys)-1 , max(ys)+1])
            hold off;
            getframe;
            
        end
        
    end
    
end
        
        


% output
x_out       = reshape(x_cur , size(y));




%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Auxiliary functions called within this function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    %--------------------%
    %----------------------------------------%
    %------------------------------------------------------------%
    function [cl, cr]  = initialize(prior, Samp_Grid, n)
    % INITIALIZE messages of the algorithm.
        
        % Number of samples
        N       = length(Samp_Grid);
        
        % Left constraints
        cl      = ones(n, N);
        cl(1, :) = priordist(Samp_Grid, prior);
        
        % Right constraints
        cr      = ones(n, N);
        
    end
    %------------------------------------------------------------%
    %----------------------------------------%
    %--------------------%
    
    


    %--------------------%
    %----------------------------------------%
    %------------------------------------------------------------%
    function p = priordist(Samp_Grid, prior)
    % PRIORDIST generates sampled prior distribution
        
        switch(prior.type)
            case{'g', 'gaussian'}
                p = gaussian(Samp_Grid, 0, prior.param);
            case{'l', 'laplace'}
                p = laplace(Samp_Grid, prior.param);
            case{'s', 'student'}
                p = student(Samp_Grid, prior.param);
            case{'a', 'alpha_stable'}
                p = alpha(Samp_Grid, 1 , prior.param);
                
            case{'pg', 'poisson-gaussian'}
                p = poissonMix('g', Samp_Grid, prior);
            case{'pl', 'poisson-laplace'}
                p = poissonMix('l', Samp_Grid, prior);
            case{'ps', 'poisson-student'}
                p = poissonMix('s', Samp_Grid, prior);
            case{'pa', 'poisson-alpha_stable'}
                p = poissonMix('a', Samp_Grid, prior);
                
            otherwise
                error('messagePassingTime: Prior not supported!');
        end
        
    end
    %------------------------------------------------------------%
    %----------------------------------------%
    %--------------------%
    
    


    %--------------------%
    %----------------------------------------%
    %------------------------------------------------------------%
    function phi = poissonMix(type, Samp_Grid, prior)
    % poisson-gaussian or ...
        
        % Sampling interval
        delta   = abs(  Samp_Grid(2) - Samp_Grid(1)  );
        OSR     = 10;
        Samp_GridF  = linspace( Samp_Grid(1)-delta/2  ,  Samp_Grid(end)+delta/2  ,  OSR * length(Samp_Grid)+1 );
        deltaF  = abs( Samp_GridF(2) - Samp_GridF(1) );
        
        
        % Indices
        i       = Samp_Grid / delta;
        
        % Kronecker delta
        kdelta  = (i==0);
        
        % Non-dirac distribution
        switch(type)
            case{'g', 'gaussian'}
                p = gaussian(Samp_GridF , 0 , prior.param);
            case{'l', 'laplace'}
                p = laplace(Samp_GridF ,  prior.param);
            case{'s', 'student'}
                p = student(Samp_GridF ,  prior.param);
            case{'a', 'alpha_stable'}
                p = alpha(Samp_GridF , 1 , prior.param);
            otherwise
                error('poissonMix: Prior not supported!');
        end
        
        % Non-dirac part
        p       = p / deltaF;
        lambda  = -log( 1 - prior.rho );
        Nterms  = 20;
        pn      = [1/deltaF];
        coef    = 1;
        phi     = 0;
        for nn = 1 : Nterms
            pn  = conv(p , pn , 'same') * deltaF;
            %     sum(pn)*deltaF
            coef= coef * lambda / nn;
            phi = phi + coef * pn;
        end
        
        % sampling and adding the dirac part
        phi     = (  sum( reshape(phi(1:end-1) , OSR , length(Samp_Grid)) ) + sum( reshape(phi(2:end) , OSR , length(Samp_Grid)) )  )  * deltaF/2;
        phi     = (1-prior.rho) * (kdelta + phi);
        % phi     = phi / sum(phi);
        
    end
    %------------------------------------------------------------%
    %----------------------------------------%
    %--------------------%




    %--------------------%
    %----------------------------------------%
    %------------------------------------------------------------%
    function phi = gaussian(Samp_Grid, mu, va)
    % GAUSSIAN generates sampled gaussian distribution
        
        % Sampling interval
        delta   = abs(Samp_Grid(2) - Samp_Grid(1));
        
        % Indices
        i       = Samp_Grid / delta;
        
        % Standard deviation
        s       = sqrt(va);
        
        % Sample from Gaussian
        phi     = normcdf(delta * (i+0.5)  ,  mu  ,  s) - normcdf(delta * (i-0.5) , mu , s);
        phi     = phi / sum(phi);
        
    end
    %------------------------------------------------------------%
    %----------------------------------------%
    %--------------------%
    
    


    %--------------------%
    %----------------------------------------%
    %------------------------------------------------------------%
    function phi = laplace(Samp_Grid, lambda)
    % LAPLACE generates sampled laplace distribution
        
        % Sample from Laplace
        phi     = 0.5 * lambda*exp(-lambda * abs(Samp_Grid));
        phi     = phi / sum(phi);
        
    end
    %------------------------------------------------------------%
    %----------------------------------------%
    %--------------------%
    
    


    %--------------------%
    %----------------------------------------%
    %------------------------------------------------------------%
    function phi = student(Samp_Grid, q)
    % STUDENT generates sampled student's-t distribution
        
        % Sampling interval
        delta   = abs(Samp_Grid(2) - Samp_Grid(1));
        
        % Indices
        i       = Samp_Grid / delta;
        
        % Sample from Student's-t
        phi     = cdf('t' , delta * (i+0.5) , q) - cdf('t' , delta * (i-0.5) , q);
        phi     = phi / sum(phi);
        
    end
    %------------------------------------------------------------%
    %----------------------------------------%
    %--------------------%


    
    %--------------------%
    %----------------------------------------%
    %------------------------------------------------------------%
    function phi = alpha(Samp_Grid, sigma , alpha)
    % ALPHA generates sampled alpha-stable distribution
        
        % Sampling interval
        delta   = abs(Samp_Grid(2) - Samp_Grid(1));
        i       = Samp_Grid / delta;    % Indices
        
        if alpha == 1   % cauchy distribution
            
            % Sample from Student's-t
            phi     = 1/pi  *  ( atan( delta * (i+0.5) / sigma ) - atan( delta * (i-0.5) / sigma ) ) ;
            phi     = phi / sum(phi);
            
        else
            
            phi     = zeros(1 , length(Samp_Grid));
            
            % Compute CDF
            V       = @(theta)    ( cos(theta) ./ sin( alpha*theta ) ) .^ (alpha / (alpha-1)) ...
                .* cos( (alpha - 1) * theta ) ./ cos(theta);
            
            Samp_GridP  = Samp_Grid(Samp_Grid > 0);
            gshiftU = ( (Samp_GridP + 0.5*delta) / sigma ) .^ ( alpha / (alpha-1) );
            gshiftL = ( (Samp_GridP - 0.5*delta) / sigma ) .^ ( alpha / (alpha-1) );
            Ppart   = sign(1-alpha)/pi * ...
                ( quadv(@(theta) exp(-gshiftU * V(theta)) , 1e-10 , pi/2-1e-10 , 1e-8) - ...
                quadv(@(theta) exp(-gshiftL * V(theta)) , 1e-10 , pi/2-1e-10 , 1e-8) );
            
            
            phi(Samp_Grid > 0)  = Ppart;
            phi(Samp_Grid < 0)  = Ppart(end:-1:1);
            phi(Samp_Grid == 0) = 2 * ( (alpha > 1) + (alpha < 1) * 0.5 + sign(1-alpha)/pi * ...
                ( quadv(@(theta) exp(- (0.5*delta/sigma) .^ ( alpha / (alpha-1) ) * V(theta)) , 1e-10 , pi/2-1e-10 , 1e-8)) ) ...
                - 1;
            
        end
        
        phi     = phi / sum(phi);
        
    end
    %------------------------------------------------------------%
    %----------------------------------------%
    %--------------------%

    
    

    %--------------------%
    %----------------------------------------%
    %------------------------------------------------------------%
    function pout = normalize(pin)
        pout    = pin / sum(pin);
    end
    %------------------------------------------------------------%
    %----------------------------------------%
    %--------------------%



    %--------------------%
    %----------------------------------------%
    %------------------------------------------------------------%
    function f = convolve(x, h)
    % CONVOLVE fast centered convolution using fftfilt.
        
        % Number of samples
        N       = length(x);
        
        % Check if N is odd
        assert(mod(N, 2) == 1, 'convolve: PDFs must have even number of samples.');
        
        % Pad-input vectos to have 2N-1 elements
        x       = [x, zeros(1, N-1)];
        h       = [h, zeros(1, N-1)];
        
        % Fast convolution
        f       = fftfilt(x, h);
        
        % Indeces to trim off
        fbeg    = (N+1)/2;
        fend    = (3*N-1)/2;
        
        % Trim
        f       = f(fbeg:fend);
    end
    %------------------------------------------------------------%
    %----------------------------------------%
    %--------------------%
    
    

    %--------------------%
    %----------------------------------------%
    %------------------------------------------------------------%
    % PROCESS_OPTIONS - Processes options passed to a Matlab function.
    %                   This function provides a simple means of
    %                   parsing attribute-value options.  Each option is
    %                   named by a unique string and is given a default
    %                   value.
    % Copyright (C) 2002 Mark A. Paskin
    function [varargout] = process_options(args, varargin)
        
        % Check the number of input arguments
        n = length(varargin);
        if (mod(n, 2))
            error('Each option must be a string/value pair.');
        end
        
        % Check the number of supplied output arguments
        if (nargout < (n / 2))
            error('Insufficient number of output arguments given');
        elseif (nargout == (n / 2))
            warn = 1;
            nout = n / 2;
        else
            warn = 0;
            nout = n / 2 + 1;
        end
        
        % Set outputs to be defaults
        varargout = cell(1, nout);
        for i=2:2:n
            varargout{i/2} = varargin{i};
        end
        
        % Now process all arguments
        nunused = 0;
        for i=1:2:length(args)
            found = 0;
            for j=1:2:n
                if strcmpi(args{i}, varargin{j})
                    varargout{(j + 1)/2} = args{i + 1};
                    found = 1;
                    break;
                end
            end
            if (~found)
                if (warn)
                    warning(sprintf('Option ''%s'' not used.', args{i}));
                    args{i}
                else
                    nunused = nunused + 1;
                    unused{2 * nunused - 1} = args{i};
                    if length(args)>i
                        unused{2 * nunused} = args{i + 1};
                    end
                end
            end
        end
        
        % Assign the unused arguments
        if (~warn)
            if (nunused)
                varargout{nout} = unused;
            else
                varargout{nout} = cell(0);
            end
        end
    end
    %------------------------------------------------------------%
    %----------------------------------------%
    %--------------------%
    
end