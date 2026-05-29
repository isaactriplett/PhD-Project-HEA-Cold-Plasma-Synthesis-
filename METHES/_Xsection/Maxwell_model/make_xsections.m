% Mohamed Rabie
% 25.11.2014
% this script makes the xsection-file for the Maxwell-model
clear all
close all
clc

energy = logspace(-10,1,1000)';
sigma = 6e-20*energy.^-(0.5);
data = [energy  sigma];

plot(energy,sigma,'b-.')
save('data.txt','data','-ascii')