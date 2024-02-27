% Clean slate
clear all, close all, clc

% Distribution parameters -beta * tan(alpha*pi/2)
alpha = 1.5; beta = 1; gam = 1; delta = 0; 

% Configuration
x = -10:0.01:10;
nrof_samples_for_histogram = 10000;

% Create new figure
figure(1)

% MATLAB framework
stabledist = makedist('Stable', 'alpha', alpha, 'beta', beta, 'gam', gam, 'delta', delta);
    matlab_pdf = pdf( stabledist, x );
    plot(x, matlab_pdf, 'k', 'LineWidth', 2); axis tight;
    leg{1} = sprintf("Matlab PDF for $\\alpha$ = %.2f, $\\beta$ = %.2f, $\\gamma$ = %.2f, $\\delta$ = %.2f", alpha, beta, gam, delta);
figure(2)
subplot(3, 1, 1)
    tic
    matlab_random_samples = random(stabledist, [nrof_samples_for_histogram, 1]);
    t_matlab = toc;
    matlab_random_samples = matlab_random_samples(matlab_random_samples<x(end) & matlab_random_samples>x(1));
    histogram(matlab_random_samples, 'Normalization', 'pdf'); axis tight;
    hold on; plot(x, matlab_pdf, 'k', 'LineWidth', 2)
    legend("Data", leg{1}, 'Interpreter', 'latex')
    title(sprintf("Matlab histogram for $\\alpha$ = %.2f, $\\beta$ = %.2f, $\\gamma$ = %.2f, $\\delta$ = %.2f (kept %d samples)", alpha, beta, gam, delta, length(matlab_random_samples)), 'Interpreter', 'latex')
    fprintf("\nMATLAB: Generated %d samples in %f s.\n", nrof_samples_for_histogram, t_matlab)
    
% External library by Mark Veillette, Ph.D. (now technical staff at MIT
% Lincoln Laboratory)
figure(1)
    veillette_pdf = stblpdf(x, alpha, beta, gam, delta);
    hold on; plot(x, veillette_pdf, 'LineWidth', 2); axis tight;
    leg{2} = sprintf("Veillette PDF for $\\alpha$ = %.2f, $\\beta$ = %.2f, $\\gamma$ = %.2f, $\\delta$ = %.2f", alpha, beta, gam, delta);
    legend(leg, 'Interpreter', 'latex')
    title("PDFs")
figure(2)
subplot(3, 1, 2)
    tic
    veillette_random_samples = stblrnd( alpha, beta, gam, delta, [nrof_samples_for_histogram, 1] );
    t_veillette = toc;
    veillette_random_samples = veillette_random_samples(veillette_random_samples<x(end) & veillette_random_samples>x(1));
    histogram(veillette_random_samples, 'Normalization', 'pdf'); axis tight;
    hold on; plot(x, veillette_pdf, 'r', 'LineWidth', 2)
    legend("Data", leg{2}, 'Interpreter', 'latex')
    title(sprintf("Veillette histogram for $\\alpha$ = %.2f, $\\beta$ = %.2f, $\\gamma$ = %.2f, $\\delta$ = %.2f (kept %d samples)", alpha, beta, gam, delta, length(veillette_random_samples)), 'Interpreter', 'latex')
    fprintf("Veillette: Generated %d samples in %f s.\n", nrof_samples_for_histogram, t_veillette)
    
% Code by Pakshal Bohra
params = [alpha, beta, gam, delta];
subplot(3, 1, 3)
    tic
    bohra_random_samples = generate_alpha_stable_rv(params, nrof_samples_for_histogram);
    t_bohra = toc;
    bohra_random_samples = bohra_random_samples(bohra_random_samples<x(end) & bohra_random_samples>x(1));
    histogram(bohra_random_samples, 'Normalization', 'pdf'); axis tight;
    hold on; plot(x, veillette_pdf, 'r', 'LineWidth', 2)
    legend("Data", leg{2}, 'Interpreter', 'latex')
    title(sprintf("Bohra histogram for $\\alpha$ = %.2f, $\\beta$ = %.2f, $\\gamma$ = %.2f, $\\delta$ = %.2f (kept %d samples)", alpha, beta, gam, delta, length(bohra_random_samples)), 'Interpreter', 'latex')
    fprintf("Bohra: Generated %d samples in %f s.\n", nrof_samples_for_histogram, t_bohra)

fprintf("------\nBy exploration: Bohra and Veillette agree, MATLAB does not. Each is internally consistent.\nMATLAB documentation (run doc StableDist) reveals weird parameterization with dead reference (maybe what is referred as M parametrization elsewhere?). For beta=0 it may always match.\n-------\n")

%% Time analysis
fprintf("Starting propper time analysis...\n") 

% Parameters
nrof_reps = 1000;
nrsof_samples = 1000:1000:11000;
nrsof_pdf_points = 10:100:1100;

% Outputs
MATLAB_gen_times = zeros([length(nrsof_samples), 1]);
veillette_gen_times = zeros([length(nrsof_samples), 1]);
bohra_gen_times = zeros([length(nrsof_samples), 1]);

MATLAB_pdf_times = zeros([length(nrsof_pdf_points), 1]);
veillette_pdf_times = zeros([length(nrsof_pdf_points), 1]);

fprintf("\tTiming random variate generation.\n\t\tSamples: ") 
for idx = 1:length(nrsof_samples)
    nrof_samples = nrsof_samples(idx);
    fprintf("%d, ", nrof_samples)
    for idc = 1:nrof_reps
        % Get random parameters (standard)
        alpha = 0.01 + 1.99*rand; beta = -1 + 2*rand; gamma = 1; delta = 0;
        % MATLAB generation
        stabledist = makedist('Stable', 'alpha', alpha, 'beta', beta, 'gam', gam, 'delta', delta);
        tic;
        random(stabledist, [nrof_samples, 1]);
        MATLAB_gen_times(idx) = MATLAB_gen_times(idx) + toc;
        % Veillette generation
        tic;
        stblrnd( alpha, beta, gam, delta, [nrof_samples, 1] );
        veillette_gen_times(idx) = veillette_gen_times(idx) + toc;
        % Bohra generation
        params = [alpha, beta, gam, delta];
        tic;
        generate_alpha_stable_rv(params, nrof_samples);
        bohra_gen_times(idx) = bohra_gen_times(idx) + toc;
    end
end
fprintf("Done.\n");
MATLAB_gen_times = MATLAB_gen_times/nrof_reps; veillette_gen_times = veillette_gen_times/nrof_reps; bohra_gen_times = bohra_gen_times/nrof_reps;

fprintf("\tTiming PDF evaluation.\n\t\tPoints: ")
for idx = 1:length(nrsof_pdf_points)
    nrof_pdf_points = nrsof_pdf_points(idx);
    fprintf("%d, ", nrof_pdf_points)
    % Construct grid
    x = -10:20/(nrof_pdf_points-1):10;
    for idc = 1:nrof_reps
        % Get random parameters
        alpha = 0.01 + 1.99*rand; beta = -1 + 2*rand; gamma = 1; delta = 0;
        % MATLAB pdf
        stabledist = makedist('Stable', 'alpha', alpha, 'beta', beta, 'gam', gam, 'delta', delta);
        tic;
        pdf(stabledist, x);
        MATLAB_pdf_times(idx) = MATLAB_pdf_times(idx) + toc;
        % Veillette pdf
        tic;
        stblpdf(x, alpha, beta, gam, delta);
        veillette_pdf_times(idx) = veillette_pdf_times(idx) + toc;
    end
end
fprintf("Done.\n");
MATLAB_pdf_times = MATLAB_pdf_times/nrof_reps; veillette_pdf_times = veillette_pdf_times/nrof_reps;

figure(3)
subplot(2, 1, 1)
    plot(nrsof_samples, MATLAB_gen_times, nrsof_samples, veillette_gen_times, nrsof_samples, bohra_gen_times, 'LineWidth', 2)
    legend("MATLAB", "Veillette", "Bohra"); title("Generation of random stable points"); xlabel("Number of points"); ylabel("Average time [s]")
subplot(2, 1, 2)
    plot(nrsof_pdf_points, MATLAB_pdf_times, nrsof_pdf_points, veillette_pdf_times, 'LineWidth', 2)
    legend("MATLAB", "Veillette"); title("Evaluation of PDF on a grid"); xlabel("Number of grid points"); ylabel("Average time [s]")
savefig("Timing_MATLAB_vs_Veillette-Bohra.fig")