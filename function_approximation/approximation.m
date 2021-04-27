% A program for approximating some function

% [x,y]=simplefit_dataset;
load("datapoints.mat");

% create a structure for our network
% training method - Levenberg-Marquardt 
net=fitnet(150);

% net.divideFcn='dividerand'; % náhodné rozdelenie
% net.divideFcn='divideblock'; % blokove

% data division - by indexes (which are also in the data we loaded)

net.divideFcn='divideind';
net.divideParam.trainInd = indx_train;
net.divideParam.testInd = indx_test;


% net.divideFcn='divideind';
% net.divideParam.trainInd=1:2:n;
% net.divideParam.valInd=2:2:n2;
% net.divideParam.testInd=n2+1:2:n;


% training parameters
net.trainParam.goal = 1e-4;     % end condition - SSE.
net.trainParam.show = 5;

% network training
net=train(net,x,y);

t = y;
y_ = net(x);

SSE = sse(net, t, y_);
MSE = mse(net, t, y_);
MAE = mae(net, t, y_);

fprintf("SSE = %f, MSE = %f, MAE = %f\n", SSE, MSE, MAE);

outnetsim = sim(net,x);

figure
x_test = x(sort(indx_test));
y_test = y(sort(indx_test));
x_train = x(sort(indx_train));
y_train = y(sort(indx_train));


plot(x_test, y_test,'^b', x_train, y_train, '^g', x, outnetsim, '-or');
legend('test data', 'train data', 'network output');