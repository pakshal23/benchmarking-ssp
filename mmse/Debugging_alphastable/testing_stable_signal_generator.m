% Testing generation of data using alpha-stable rv generator

K = 100;

handles1.Prior = 'student';
handles1.K = K;
handles1.Operator = 1;
handles1.Dist_Param = 1;

handles2.Prior = 'alpha-stable';
handles2.K = K;
handles2.Operator = 1;
handles2.Dist_Param = [1, 0, 1, 0];

num_signals = 5000;

x1 = cell([num_signals, 1]);
x2 = cell([num_signals, 1]);

pow1 = zeros([num_signals, 1]);
pow2 = zeros([num_signals, 1]);

for i = 1:num_signals
   
    x1{i,1} = generate_discrete_process(handles1);
    x2{i,1} = generate_discrete_process(handles2);
    
    pow1(i, 1) = norm(x1{i,1});
    pow2(i, 1) = norm(x2{i,1});
    
end