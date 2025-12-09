data = readmatrix("C:\Users\isaac\OneDrive\Documents\McGill\Townsend Approximation\data\Kruithofdata.csv");
x = data(:, 1);
y = data(:, 2);

customEq = fittype('a*exp(-(b/x))+(c*x^g)*(1-exp(-(d/x)^f))','independent','x','dependent','y')

fo = fitoptions('Method','NonlinearLeastSquares','Lower',[0,0,0,0,0,0],'StartPoint',[1,1,1,1,1,1])

fitResult = fit(x, y, customEq,fo);

disp(fitResult);


figure;
plot(fitResult,x,y)
xlabel('x (V/(cm mmHg)')
ylabel('z (cm^{-1} mmHg^{-1})');
title(['Kruithof and Penning Data and Fit'])
legend('Location','southeast')
