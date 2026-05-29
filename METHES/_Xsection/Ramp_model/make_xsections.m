% Mohamed Rabie
% 25.11.2014
% this script makes the xsection-file for the Ramp-model
clear all
close all
clc
A = 10^-10;
energy = [0 : 0.1 : 10]';

sigma = 6*A^2*ones(size(energy));
data = [energy  sigma];

plot(energy,sigma,'b-.')
save('elastic.txt','data','-ascii')

sigma = max(0,10*A^2*(energy-0.2)); 

data = [energy  sigma];
hold on
plot(energy,sigma,'r-.')
save('inelastic.txt','data','-ascii')