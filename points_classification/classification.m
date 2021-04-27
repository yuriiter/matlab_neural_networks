
load("datapoints.mat");


% data for class 1
xA = data1(:, 1).';
yA = data1(:, 2).';
zA = data1(:, 3).';

% data for class 2
xB = data2(:, 1).';
yB = data2(:, 2).';
zB = data2(:, 3).';

% data for class 3
xC = data3(:, 1).';
yC = data3(:, 2).';
zC = data3(:, 3).';

% data for class 4
xD = data4(:, 1).';
yD = data4(:, 2).';
zD = data4(:, 3).';

% data for class 5
xE = data5(:, 1).';
yE = data5(:, 2).';
zE = data5(:, 3).';



% plot the points of the classes
h=figure;

hold on
plot3(xA, yA, zA, 'o');
plot3(xB, yB, zB, 'o');
plot3(xC, yC, zC, 'o');
plot3(xD, yD, zD, 'o');
plot3(xE, yE, zE, 'o');
xlabel('x');
ylabel('y');
zlabel('z');


% input data for the network
V1=[xA xB xC xD xE];
V2=[yA yB yC yD yE];
V3=[zA zB zC zD zE];
X=[V1;V2;V3];

% targets
P=[ones(1, 50) zeros(1, 50) zeros(1, 50) zeros(1, 50) zeros(1, 50);...
    zeros(1, 50) ones(1, 50) zeros(1, 50) zeros(1, 50) zeros(1, 50);...
    zeros(1, 50) zeros(1, 50) ones(1, 50) zeros(1, 50) zeros(1, 50);...
    zeros(1, 50) zeros(1, 50) zeros(1, 50) ones(1, 50) zeros(1, 50);...
    zeros(1, 50) zeros(1, 50) zeros(1, 50) zeros(1, 50) ones(1, 50)];


% creating a structure for our classification network
net = patternnet(11);
net.trainFcn = 'trainbr';

% random division of data for training and testing
net.divideFcn='dividerand';
net.divideParam.trainRatio = 0.8;
% net.divideParam.valRatio = 0.1;
net.divideParam.testRatio = 0.2;

net.performFcn = 'sse';
% net.performParam.regularization = 0.1;

net.trainParam.goal = 1e-10;        % end condition for SSE.
net.trainParam.epochs = 300;        % maximal number of epochs
net.trainParam.min_grad = 1e-100;   % end condition for min. gradient


% network training

% size(input)  = [ I N ]
% size(target) = [ O N ]

net = train(net,X,P);

% simulate network output
y = net(X);
% network perfomance
perf = perform(net,P,y);

% assign inputs to the classes
classes = vec2ind(y);

% test with customly chosen points
X2 = [0.494407508582969 0.596112590406635 0.882758620689655 0.450793650793651 0.570469798657718;...
      0.247817684966992 0.259035360066034 0.905826521590068 0.771411128553986 0.731043751177979;...
      0.580520834211215 0.125848428349602 0.562500000000000 0.650735294117647 0.444852941176471];

y2 = net(X2);

% assign inputs to the classes
classes2 = vec2ind(y2);
% targets = [1 0 0 0 0;...
%            0 1 0 0 0;...
%            0 0 1 0 0;...
%            0 0 0 1 0;...
%            0 0 0 0 1];

% performance = perform(net, targets, y2);
% fprintf("%s\n", performance);
for i = 1 : 5
   fprintf("Point {%.2f, %.2f, %.2f} is of class %d\n", X2(1,i), X2(2,i), X2(3,i), classes2(i));
   plot3(X2(1,i), X2(2,i), X2(3,i), "*");
end
