% run_methes_Ar_sweep.m
% Batch E/N sweep for pure argon (METHES) -> transport tables for afivo-streamer.
% Faithful to MonteCarlo_single_run.m, run headless, one file saved per E/N point.
% Put this under the METHES root, e.g. .../METHES/Ar_sweep/, and run MATLAB
% with that folder as the current directory so '..\_functions' resolves.

clear all; close all; clc;

%% ===== USER SETTINGS =====================================================
functionsDir = '..\_functions';
gasDir       = {'..\_Xsection\Ar_Biagi'};   % argon WITH ionization
gas          = {'Ar'};
mix          = 1;                            % pure Ar
outDir       = '.\results_Ar';

% E/N grid in Townsend
% Route A: METHES handles E/N >= 10 Td; BOLSIG+ fills below.
EN_log  = logspace(log10(0.1), log10(1000), 36);
EN_fine = 30:20:400;
EN_list = unique(round([EN_log, EN_fine], 3));
EN_list = EN_list(EN_list >= 10);     % -> 37 points, 11.4 ... 1000 Td
%EN_list=100 %for testing

% Background (does NOT affect reduced swarm outputs; only MC time scaling)
p    = 1e5;    % Pa
Temp = 300;    % K

% Monte Carlo controls (your stock defaults)
N0=1e4; Ne_max=1e6; W=0.5; w_err=0.01; DN_err=0.01;
col_equ=1e7; col_max=1e8; conserve=1;
interactive = 0;       % MUST be 0: no plots, no pauses
%% ========================================================================

addpath(functionsDir);
if ~exist(outDir,'dir'); mkdir(outDir); end

% Import cross sections ONCE (independent of E/N)
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
% (plotXsections deliberately omitted for headless running)

nEN = numel(EN_list);
fprintf('METHES Ar sweep: %d E/N points -> %s\n', nEN, outDir);

for i = 1:nEN
    EN = EN_list(i);
    outFile = fullfile(outDir, sprintf('Ar_EN_%08.3fTd.mat', EN));
    if exist(outFile,'file')
        fprintf('[%3d/%3d] %8.3f Td  done, skip\n', i, nEN, EN); continue;
    end

    if     EN<=1,   energy=0:0.002:30;
    elseif EN<=10,  energy=0:0.005:60;
    elseif EN<=100, energy=0:0.01:100;
    elseif EN<=300, energy=0:0.02:200;
    elseif EN<=600, energy=0:0.05:400;
    else            energy=0:0.10:600;
    end
    fprintf('[%3d/%3d] %8.3f Td (Emax=%g eV)\n', i, nEN, EN, energy(end));

    % --- build swarm object exactly as in your stock script ---------------
    sig = MonteCarlo;
    sig.Xsec=Xsec; sig.gas=gas; sig.mix=mix; sig=sig.mass_in_kg();
    sig.N0=N0; sig.Ne_max=Ne_max; sig.p=p; sig.Temp=Temp; sig.EN=EN;
    sig.E_max=energy(end); sig.E.energy=energy; sig.W=W; sig.iso=1;
    sig.w_err=w_err; sig.DN_err=DN_err; sig.col_equ=col_equ;
    sig.col_max=col_max; sig.conserve=conserve; sig.interactive=interactive;
    sig.sigma_xyz=[0 0 0]; sig.pos_xyz=[0 0 0];
    sig=sig.checkFractionSum(); sig=sig.gasNumberDensity();
    sig=sig.maximalCollFreq(); sig=sig.solvePoisson_3D(100,0);
    sig=sig.initialParticles();
    sig.counter=0; sig.flux.v_int_sum=[0 0 0]; sig.flux.D_sum=0;
    sig.flux.N=0; sig.E.E_sum=0; sig.E.EEPF_sum=0;

    % --- steady-state loop (identical method sequence to stock) -----------
    while sig.End == 0
        sig=sig.freeFlight();      sig=sig.collectMeanData();
        sig=sig.energyData();      sig=sig.fluxData();
        sig=sig.bulkData();        sig=sig.rateDataCount();
        sig=sig.rateDataConv();    sig=sig.collisionMatrix();
        sig=sig.elasticCollision();sig=sig.inelasticCollision();
        sig=sig.ionCollision();    sig=sig.attachCollision();
        sig=sig.plotMeanData();    sig=sig.checkSST();
        sig=sig.printOnScreen();   sig=sig.endSimulation();
    end

    % --- save this point --------------------------------------------------
    if exist('results.mat','file')
        movefile('results.mat', outFile);     % native METHES output
    else
        res.EN=EN; res.gas=gas; res.mix=mix; res.p=p; res.Temp=Temp;
        res.energy=energy; res.E=sig.E; res.flux=sig.flux; res.bulk=sig.bulk;
        try, res.conv=sig.conv; end
        try, res.count=sig.count; end
        save(outFile,'res','-v7');
    end
    clear sig;
end
fprintf('Sweep complete.\n');