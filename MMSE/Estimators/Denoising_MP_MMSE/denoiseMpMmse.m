function xhat = denoiseMpMmse(y,handles)
% Wrapper file for message passing algorithm

if handles.Prior_Type == 1
    switch handles.Dist_Sel
        case{1} 
            prior.type = 'g';
        case{2}
            prior.type = 'l';
        case{3}
            prior.type = 's';
        case{4}
            prior.type = 'a';
    end 
elseif handles.Prior_Type == 2
    prior.type = 'pl';
    prior.rho  = 1-handles.MassProb;
else
    error('Unknown signal type!');
end

% Sampling grid
xmax  = ceil(max(abs(y))+2*sqrt(handles.Dist_Param+handles.Noise_Var));
delta = min(handles.Dist_Param,handles.Noise_Var);
imax  = ceil(xmax/delta - 0.5);
sgrid = (-imax : imax) * delta;

prior.param = handles.Dist_Param;

% Compute MP estimate
xhat = TMP_MMSE_Levy(y,sgrid,prior,handles.Noise_Var);
end