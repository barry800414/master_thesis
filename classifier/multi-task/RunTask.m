

clear, clc;
addpath('/home/r02922010/package/MALSAR/MALSAR/utils/')
addpath('/home/r02922010/package/MALSAR/MALSAR/functions/Lasso/')
addpath('/home/r02922010/package/MALSAR/MALSAR/functions/joint_feature_learning/')
addpath('/home/r02922010/package/MALSAR/MALSAR/functions/low_rank/')

% setting
seed = 1;
foldNum = 10;
p1Range = 2.^(-17:2:5);
method = 'Logistic_Lasso';
opts.init = 0;      % guess start point from data.
opts.tFlag = 1;     % terminate after relative objective value does not changes much.
opts.tol = 10^-6;   % tolerance.
opts.maxIter = 1500; % maximum iteration number of optimization.

% read in files: X, Y of t task
load('data.mat');

% generate 10-fold training and testing data
rng(seed);
taskNum = length(Y);
dim = size(X{1}); dim = dim(2);
XTrain = cell(taskNum, 1);
YTrain = cell(taskNum, 1);
XTest = cell(taskNum, 1);
YTest = cell(taskNum, 1);
kfold = cell(foldNum, 1);
taskProp = zeros(1, taskNum); % proportion of data of each task
for t=1:taskNum
    taskProp(t) = length(Y{t});
    kfold{t} = crossvalind('KFold', Y{t}, foldNum);
end
taskProp = taskProp / sum(taskProp);

% print first line of result
fprintf('framework, classifier, scorer, dimension, randSeed, foldNum, train, val, test');
for t=1:taskNum
    fprintf(', train_%d, val_%d, test_%d', t, t, t); 
end
fprintf('\n');

% 10-fold testing 
for fi=1:foldNum
    fprintf(2, 'Running Fold %d ... \n', fi);
    % get training and testing data of each task
    for t=1:taskNum
        XTrain{t} = X{t}(kfold{t} ~= fi, :);
        XTest{t} = X{t}(kfold{t} == fi, :);
        YTrain{t} = Y{t}(kfold{t} ~= fi, :);
        YTest{t} = Y{t}(kfold{t} == fi, :);
    end
    % using cross-validation to search best parameter
    [ p1, valAcc ] = GridSearchCV( XTrain, YTrain, ...
        foldNum, seed, method, p1Range, opts );
    % testing on testing set
    [ W, c, YTrainPredict, trainAcc, YTestPredict, testAcc ] = TrainTest( ...
        XTrain, YTrain, XTest, YTest, method, p1, opts);
 
    weightedTrainAcc = taskProp * trainAcc ;
    weightedValAcc = taskProp * valAcc ;
    weightedTestAcc = taskProp * testAcc ;
    fprintf('MultiTask, %s, Accuracy, %d, %d, %d, %f, %f, %f', method, ...
        dim, seed, foldNum, weightedTrainAcc, weightedValAcc, weightedTestAcc);
    for t=1:taskNum
        fprintf(', %f, %f, %f', trainAcc(t), valAcc(t), testAcc(t));
    end
    fprintf('\n');
end
