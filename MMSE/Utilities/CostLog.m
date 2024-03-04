classdef CostLog < Cost
    % CostLog: Log cost function (potential function for the Student's t-distribution) 
    % C(x) = log(x^2 + epsilon)
    %
    % All attributes of parent class :class:`Cost` are inherited. 
    %
    % :param epsilon: scalar (default 1)
    % :param prox_op: implementation of the proximal operator of
    % \frac{\lambda}{\rho} C(x)
    %
    % **Example** C=CostLog(sz, epsilon, prox_op)
    %
    % See also :class:`Map`, :class:`Cost`, :class:`LinOp`

    % Protected Set and public Read properties
    properties (SetAccess = protected,GetAccess = public)
        epsilon = 1;
        prox_op = @(x) x;
    end

    %% Constructor
    methods
        function this = CostLog(sz, epsilon, prox_op)
            y=0; 
            this@Cost(sz,y);
            this.epsilon = epsilon;
            this.prox_op = prox_op;
            this.name='CostLog';
%             this.isConvex=false;
%             this.isSeparable=true;
%             this.isDifferentiable=false;
        end        
    end
    
    %% Core Methods containing implementations (Protected)
    % - apply_(this,x)
    % - applyProx_(this,x,alpha)

	methods (Access = protected)
        function y=apply_(this,x)
        	% Reimplemented from parent class :class:`Cost`.       
            if (isscalar(this.y)&&(this.y==0))
                y = sum(log(this.epsilon + (abs(x)).^2));
            else
                y = sum(log(this.epsilon + (abs(x)).^2));
            end
        end

        
        function y=applyProx_(this,x,alpha)
        	% Reimplemented from parent class :class:`Cost`   
            % $$ \\mathrm{prox}_{\\alpha C}(\\mathrm{x}) = \\frac{\\mathrm{x} + \\alpha \\mathrm{Wy}}{1 + \\alpha \\mathrm{W}} $$
            % where the division is component-wise.
            
            y = this.prox_op(x); % \alpha (=lambda/rho) should be included in the LUT for the prox.
                
        end

    end
end
