
function [ avgTestAcc ] = OneTestSingleFold( dataFile, method, topic, seed, foldId, outFilePrefix )
% topic: from 1 to 4

LoadPackage();

% setting
foldNum = 10;
pRange = ParamRange(method);
opts.init = 0;      % guess start point from data.
opts.tFlag = 1;     % terminate after relative objective value does not changes much.
opts.tol = 10^-6;   % tolerance.
opts.maxIter = 1500; % maximum iteration number of optimization.

% read in files: X, Y of t task
load(dataFile);

% open result file to write
fout = fopen(strcat(outFilePrefix, '_result.csv'), 'w');
taskNum = length(Y);

% print first line of result
if seed == 1 && foldId == 1
    fprintf(fout, 'framework, classifier, scorer, dimension, randSeed, foldNum, train, val, test\n');
end

% generate training and testing data (only 10% of one topic is testing data
rng(seed);
dim = size(X{1}); dim = dim(2);

XTrain = cell(taskNum, 1);
YTrain = cell(taskNum, 1);
XTest = cell(taskNum, 1);
YTest = cell(taskNum, 1);
kfold = crossvalind('KFold', Y{topic}, foldNum);
fi = foldId; 

fprintf(2, 'Running Fold %d ... \n', fi);
% get training and testing data of each task
for t=1:taskNum
    if t == topic
        XTrain{t} = X{t}(kfold ~= fi, :);
        XTest{t} = X{t}(kfold == fi, :);
        YTrain{t} = Y{t}(kfold ~= fi, :);
        YTest{t} = Y{t}(kfold == fi, :);
    else
        XTrain{t} = X{t};
        XTest{t} = [];
        YTrain{t} = Y{t};
        YTest{t} = [];
    end
end

% using cross-validation to search best parameter 
% (p(1) to p(N) are for individual, p(N+1) are for all mixed)
[ params, valAcc ] = OneTestGridSearchCV( XTrain, YTrain, topic, foldNum, seed, method, pRange, opts );

PrintParam(params);
% testing on testing set
[ model, YTrainPredict, trainAcc, YTestPredict, testAcc ] = TrainTestSingleTask( ...
    XTrain, YTrain, XTest, YTest, method, params, opts, topic);

fprintf(fout, 'MultiTask, %s, Accuracy, %d, %d, %d, %f, %f, %f', method, ...
    dim, seed, foldNum, trainAcc, valAcc, testAcc);
fprintf(fout, '\n');
fclose(fout);

save(strcat(outFilePrefix, '.mat'), 'dataFile', 'method', 'topic', 'seed', 'foldId', ...
    'params', 'model', 'trainAcc', 'YTrainPredict', 'valAcc', 'testAcc', 'YTestPredict');

end
