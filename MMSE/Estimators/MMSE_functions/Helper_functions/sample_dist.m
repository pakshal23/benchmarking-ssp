function [samp] = sample_dist(p1, p2, p3, m1, m2, var_i)

cat_samp_aux = rand();

if (cat_samp_aux < p1)
    cat_samp = 1;
    
elseif (cat_samp_aux < p1 + p2)
    cat_samp = 2;
        
else
    cat_samp = 3;
    
end


if (cat_samp == 1)
    samp = 0;
    
elseif (cat_samp == 2)
    pd = makedist('Normal','mu',m1,'sigma',sqrt(var_i));
    t = truncate(pd,0,inf);
    samp = random(t,1);
    
else
    pd = makedist('Normal','mu',m2,'sigma',sqrt(var_i));
    t = truncate(pd,-inf,0);
    samp = random(t,1);


end

