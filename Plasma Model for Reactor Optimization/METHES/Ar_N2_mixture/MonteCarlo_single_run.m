%     Copyright (c) 2015, Mohamed Rabie
%     All rights reserved.
%     
%     Redistribution and use in source and binary forms, with or without
%     modification, are permitted provided that the following conditions are
%     met:
%     - Redistributions of source code must retain the above copyright notice, 
%     this list of conditions and the following disclaimer.
%     - Redistributions in binary form must reproduce the above copyright notice, 
%     this list of conditions and the following disclaimer in the documentation 
%     and/or other materials provided with the distribution
%     - Proper reference is made in publications reporting results obtained 
%     using this software. At present, the preferred way to reference METHES is as follows: 
%     M. Rabie and C.M. Franck, "A Monte Carlo collision code for electron transport in low temperature plasmas", 
%     to be submitted to J. Phys. D: Appl. Phys. (2015) 
%     as well as M. Rabie and C.M. Franck, METHES code, downloaded from www.lxcat.net/download/METHES/, 2015
%         
%     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
%     AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
%     IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
%     ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
%     LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
%     CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
%     SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
%     INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
%     CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
%     ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
%     POSSIBILITY OF SUCH DAMAGE.

clear all
close all
clc

%% INPUT
functionsDir = '..\_functions';
addpath(functionsDir);

% directories of cross sections
gasDir = {'..\_Xsection\Ar_Biagi',...
    '..\_Xsection\N2_Biagi',...
    };
% sum formula gases
gas = {'Ar','N2'};
% mixing ratio
mix = [0.3 0.7]; % 30% Ar and 70% N2
% E/N in Townsend
EN = 100;
% pressure in Pascal
p = 1e6;
% temperature in Kelvin
Temp = 300;
% start electron number
N0 = 1e4;
% maximum allowed electron number
Ne_max = 1e6;
% energy sharing factor in interval [0,1]
W = 0.5;
% energy vector
energy = 0:0.01:100;
% error tolerance for drift velocity
w_err = 0.01;
% error tolerance for diffusion constant
DN_err = 0.01;
% minimum number of collisions before steady-state
col_equ = 1e7;
% maximum number of collisions of simulation
col_max = 1e8;
% conserve electron number (1) or not(0)
conserve = 1;
% plot data (1) or not(0)
interactive  = 1;
%%

%% import cross sections
%%
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

%% PLOTS
textsize = 14;
xscale = 'log' ; yscale = 'log';
Xsec.plotXsections(textsize,xscale,yscale);
%%

sig = MonteCarlo;
sig.Xsec = Xsec;
sig.gas = gas;
% mixing ratio of gases
sig.mix = mix;
% molecular mass of gases in kg
sig = sig.mass_in_kg();

% conditions:
sig.N0 = N0 ; % number of initial electrons
sig.Ne_max = Ne_max ; % maximum number of electrons
sig.p = p; % Pressure in Pascal
sig.Temp = Temp; % Temperature in Kelvin
sig.EN = EN; % in Td

% numerics
sig.E_max = energy(end);
sig.E.energy = energy;
sig.W = W;
sig.iso = 1;
sig.w_err = w_err;
sig.DN_err = DN_err;
sig.col_equ = col_equ;
sig.col_max = col_max;
sig.conserve = conserve; % conserve (1) electron number or not (0)
sig.interactive = interactive; % plot (1) or do not plot data (0)
%%

% initial electrons
sig.sigma_xyz = [0 0 0];
sig.pos_xyz = [0 0 0];

% check mixture
sig = sig.checkFractionSum();
% gas density in m^-3
sig = sig.gasNumberDensity();
% maximal collision rate
sig = sig.maximalCollFreq();
% electric field at positions x
sig = sig.solvePoisson_3D(100,0);
% set initial electron position and velocity
sig = sig.initialParticles();

sig.counter = 0;
sig.flux.v_int_sum = [0 0 0];
sig.flux.D_sum = 0;
sig.flux.N = 0;
sig.E.E_sum = 0;
sig.E.EEPF_sum = 0;

tic
while sig.End == 0
    
    % perform a flight for all electrons without collisions
    sig = sig.freeFlight();
    % calculates the collective data of electron swarm
    sig = sig.collectMeanData();
    % get mean energy and EEDF
    sig = sig.energyData();
    % get flux transport coefficients
    sig = sig.fluxData();
    % get bulk transport coefficients
    sig = sig.bulkData();
    % get reaction rates from electron number
    sig = sig.rateDataCount();
    % get reaction rates from convolution with EEDF
    sig = sig.rateDataConv();
    % decides which type of collision will happen
    sig = sig.collisionMatrix();
    % perform elastic collisions
    sig = sig.elasticCollision();
    % perform inelastic collisions
    sig = sig.inelasticCollision();
    % perform ionization collisions
    sig = sig.ionCollision();
    % perform attachment collisions
    sig = sig.attachCollision();
    % plot collective data
    sig = sig.plotMeanData();
    % check if equilibrium is reached
    sig = sig.checkSST();
    % print results in command window
    sig = sig.printOnScreen();
    % end simulation
    sig = sig.endSimulation();
    
end

