
function [ avgTestAcc ] = RunTask( dataFile, method, seed, foldId, outFilePrefix )

LoadPackage();

% setting
%seed = 1;
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

% print first line of result
taskNum = length(Y);
if seed == 1 && foldId == 1
    fprintf(fout, 'framework, classifier, scorer, dimension, randSeed, foldNum, train, val, test');
    for t=1:taskNum
        fprintf(fout, ', train_%d, val_%d, test_%d', t, t, t); 
    end
    fprintf(fout, '\n');
end

% generate 10-fold training and testing data
rng(seed);
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

fi = foldId;
% 10-fold testing 
fprintf(2, 'Running Fold %d ... \n', fi);
% get training and testing data of each task
for t=1:taskNum
    XTrain{t} = X{t}(kfold{t} ~= fi, :);
    XTest{t} = X{t}(kfold{t} == fi, :);
    YTrain{t} = Y{t}(kfold{t} ~= fi, :);
    YTest{t} = Y{t}(kfold{t} == fi, :);
end
% using cross-validation to search best parameter
[ p, valAcc ] = GridSearchCV( XTrain, YTrain, ...
    foldNum, seed, method, pRange, opts );
PrintParam(p);
% testing on testing set
[ model, YTrainPredict, trainAcc, YTestPredict, testAcc ] = TrainTest( ...
    XTrain, YTrain, XTest, YTest, method, p, opts);
weightedTrainAcc = taskProp * trainAcc ;
weightedValAcc = taskProp * valAcc ;
weightedTestAcc = taskProp * testAcc ;
fprintf(fout, 'MultiTask, %s, Accuracy, %d, %d, %d, %f, %f, %f', method, ...
    dim, seed, foldNum, weightedTrainAcc, weightedValAcc, weightedTestAcc);
for t=1:taskNum
    fprintf(fout, ', %f, %f, %f', trainAcc(t), valAcc(t), testAcc(t));
end
fprintf(fout, '\n');
fclose(fout);

end
