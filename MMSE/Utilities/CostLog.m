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

    %%    Copyright (C) 2017 
    %     E. Soubies emmanuel.soubies@epfl.ch 
    %     F. Soulez ferreol.soulez@epfl.ch
    %
    %     This program is free software: you can redistribute it and/or modify
    %     it under the terms of the GNU General Public License as published by
    %     the Free Software Foundation, either version 3 of the License, or
    %     (at your option) any later version.
    %
    %     This program is distributed in the hope that it will be useful,
    %     but WITHOUT ANY WARRANTY; without even the implied warranty of
    %     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    %     GNU General Public License for more details.
    %
    %     You should have received a copy of the GNU General Public License
    %     along with this program.  If not, see <http://www.gnu.org/licenses/>.
    
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
