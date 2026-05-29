pH = [0:0.01:14]

function pHdependence = pHDependence(E0,slope)

pH = [0:0.01:14]
pHdependence = E0 + pH.*slope

end

figure;

hold on;

plot(pH, pHDependence(0,-0.059), 'LineWidth', 2);
xlabel('pH');
ylabel('Potential (V)');
title('pH Dependence of HER Potential');

yline(-0.277,'k--','Cobalt')
yline(-0.44,'r--','Iron')
yline(-0.7618,'g--','Zinc')
yline(-1.17,'c--','Manganese')
yline(-2.31,'b--','Aluminum')