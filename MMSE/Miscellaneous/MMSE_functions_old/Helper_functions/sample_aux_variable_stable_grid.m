function [out] = sample_aux_variable_stable_grid(increments, params)

N = length(increments);
out = zeros([N,1]);

Alpha = params(1);
stablePDF = makedist('Stable','Alpha',Alpha/2,'beta',1,'gam', (cos(pi*Alpha/4))^(2/Alpha),'delta',0);

for i = 1:N
    
    u = increments(i); 
    
    if (abs(u) < 20)
        
        grid = linspace(1e-3, 10, 1024);
        fact2 = pdf(stablePDF, grid);
        fact1 = exp( - u^2 ./ (4*grid) ) ./ sqrt(4*pi*grid);
        condPDF = fact1 .* fact2;
        out(i) = SampleViaCDF(condPDF, grid);
        
    else
        
        maxval = ceil(abs(u)*1e3);
        grid = linspace(1e-1,maxval,10*maxval);
        fact2 = pdf(stablePDF, grid);

        fact1 = exp( - u^2 ./ (4*grid) ) ./ sqrt(4*pi*grid);
        condPDF = fact1 .* fact2;
        out(i) = SampleViaCDF(condPDF, grid);
        
    end

  

    
end

end