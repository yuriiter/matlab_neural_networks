
load("data_arrhythmia.mat");

data_input = NDATA.';
data_output = typ_ochorenia.';

targets = zeros(10, 442);

for i = 1 : 442
   targets(data_output(i), i) = 1;
end


% creating a structure for classification network
net = patternnet(25);

% data division
net.divideFcn='dividerand';
net.divideParam.trainRatio = 0.6;
net.divideParam.valRatio = 0.0;
net.divideParam.testRatio = 0.4;

net.trainParam.goal = 5e-16;
net.trainParam.epochs = 700;
net.trainParam.min_grad = 1e-16;


% netowrk training

% size(input)  = [ I N ] -> [277 442]
% size(target) = [ O N ] -> [10 442]

[net, tr] = train(net, data_input, targets);

t = targets;
y = net(data_input);

trainTargets = t .* tr.trainMask{1};
valTargets = t .* tr.valMask{1};
testTargets = t .* tr.testMask{1};

figure;
plotconfusion(testTargets, y, "Test data");
figure;
plotconfusion(trainTargets, y, "Train data");
figure;
plotconfusion(targets, y);



perf = perform(net, targets, y);


[c,cm,ind,per] = confusion(trainTargets, y);
tpr_train = cm(1, 1) / sum(cm(:, 1));
tnr_train = cm(2, 2) / sum(cm(:, 2));
fnr_train = cm(1, 2) / sum(cm(:, 2));
fpr_train = cm(2, 1) / sum(cm(:, 1));

[c,cm,ind,per] = confusion(testTargets, y);
tpr_test = cm(1, 1) / sum(cm(:, 1));
tnr_test = cm(2, 2) / sum(cm(:, 2));
fnr_test = cm(1, 2) / sum(cm(:, 2));
fpr_test = cm(2, 1) / sum(cm(:, 1));


sensitivity_test = tpr_test / (tpr_test + fnr_test);
sensitivity_train = tpr_train / (tpr_train + fnr_train);

specifity_test = tnr_test / (tnr_test + fpr_test);
specifity_train = tnr_train / (tnr_train + fpr_train);


fprintf("Test data:\n\tsensitivity: %f\n\tspecifity: %f\nTrain data:\n\tsensitivity: %f\n\tspecifity: %f\n",...
    sensitivity_test, specifity_test, sensitivity_train, specifity_train);


% assign inputs to the classes
classes = vec2ind(y);



% Sensitivity TP / (TP + FN)
% Specificity TN / (TN + FP)

% Test on some data
X2 = zeros(277, 10);
targets2 = zeros(10, 10);
for i = 1 : 10
    first_found = find(classes == i);
    first_found = first_found(1);
    X2(:, i) = data_input(:, first_found);
    targets2(i, i) = 1;
end


y2 = net(X2);

classes2 = vec2ind(y2);

fprintf("%d ", classes2);
fprintf("\n");