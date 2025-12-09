%for Chapter 3 problem 13 of Herbert and Goldstein Textbook

syms t
syms theta(t)

eqn = diff(theta,t,2) - 4* ((diff(theta,t))^2)*sin(2*theta) == 0;
cond1 = theta(0) == 0;

thetaprime(t) = diff(theta,t);

cond2 = thetaprime(0) == 1;  % Non-zero initial velocity for non-trivial solution

tic  % start timer

thetaSol(t) = dsolve(eqn, [cond1, cond2],Implicit=true);

toc  % stop timer
