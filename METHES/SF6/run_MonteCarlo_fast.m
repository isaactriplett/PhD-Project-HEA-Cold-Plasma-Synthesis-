% ====================================================
%   FAST MEX Monte Carlo Runner (Close to your original)
% ====================================================

clear all
close all
clc

%% ====================== INPUT (Same as your original) ======================
functionsDir = '..\_functions_modified_for_Expected_Value';
addpath(functionsDir);

% directories of cross sections
gasDir = {'..\_Xsection\SF6_Phelps'};

% sumforumla gases
gas = {'SF6'};
% mixing ratio
mix = [1];
% E/N in Townsend
EN = 500;
% pressure in Pascal
p = 1e5;
% temperature in Kelvin
Temp = 300;
% start electron number
N0 = 1e4;
% maximum allowed electron number
Ne_max = 1e5;
% energy sharing factor in interval [0,1]
W = 0.5;
% energy vector
energy = 0:0.01:1000;
% error tolerance
w_err = 0.01;
DN_err = 0.01;
% minimum number of collisions before steady-state
col_equ = 1e7;
% maximum number of collisions of simulation
col_max = 1e8;
% conserve electron number (1) or not(0)
conserve = 0;
% plot data (1) or not(0) → 0 for speed
interactive = 1;

%% ====================== IMPORT CROSS SECTIONS ======================
Xsec = importLXcat;
Xsec.dir = gasDir;
Xsec.interactive = interactive;
Xsec = Xsec.importXsections();
Xsec = Xsec.fillThresholds();
Xsec = Xsec.getEnergy();
Xsec = Xsec.prepareForFit1();
Xsec = Xsec.momentum2elastic();
Xsec = Xsec.effective2elastic();
Xsec = Xsec.prepareForFit2();
Xsec = Xsec.totalXsection();

%% ====================== CREATE MONTECARLO OBJECT ======================
sig = MonteCarlo;

sig.Xsec = Xsec;
sig.gas = gas;
% mixing ratio of gases
sig.mix = mix;

% molecular mass of gases in kg
sig = sig.mass_in_kg();

% conditions:
sig.N0 = N0 ; 
sig.Ne_max = Ne_max ; 
sig.p = p; 
sig.Temp = Temp; 
sig.EN = EN; 

% numerics
sig.E_max = energy(end);
sig.E.energy = energy;
sig.W = W;
sig.iso = 1;
sig.w_err = w_err;
sig.DN_err = DN_err;
sig.col_equ = col_equ;
sig.col_max = col_max;
sig.conserve = conserve; 
sig.interactive = interactive; 

% initial electrons
sig.sigma_xyz = [0 0 0];
sig.pos_xyz = [0 0 0];

% check mixture
sig = sig.checkFractionSum();

% gas density
sig = sig.gasNumberDensity();

% maximal collision rate
sig = sig.maximalCollFreq();

% electric field
sig = sig.solvePoisson_3D(100,0);

% set initial electron position and velocity
sig = sig.initialParticles();

%end simulation if the right conditions are met
sig = sig.endSimulation();

% Initialize other variables
sig.counter = 0;
sig.flux.v_int_sum = [0 0 0];
sig.flux.D_sum = 0;
sig.flux.N = 0;
sig.E.E_sum = 0;
sig.E.EEPF_sum = zeros(size(sig.E.energy));
sig.line = 1;
sig.End = 0;

%% ====================== FAST MEX LOOP ======================
disp('Starting fast MEX simulation...');
tic

while sig.End == 0
    sig = monteCarlo_step_mex(sig);     % Fast compiled step
    
    % Rare plotting only if needed
    if interactive && mod(sig.counter, 5000) == 0
        sig = sig.plotMeanData();
    end
end

elapsedTime = toc;
fprintf('\nSimulation finished in %.2f seconds\n', elapsedTime);
fprintf('Total collisions: %d\n', sig.collisions);

%% ====================== SAVE RESULTS ======================
sig = sig.getResults();
disp('Results saved in "results.mat"');