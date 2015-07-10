function [ avgTestAcc ] = KFoldTrainTest( X, Y, foldNum, seed, method, p1, opts )
%TRAINVALTEST Summary of this function goes here
% bestP1: the best parameter p1 by grid search 
% bestWeightedAcc: the best weighted average at that parameter
% bestAcc: the best average of each task at that parameter (taskNum by 1 vector)
%   Detailed explanation goes here

% generate 10-fold training and testing data

taskNum = length(Y);
XTrain = cell(taskNum, 1);
YTrain = cell(taskNum, 1);
XTest = cell(taskNum, 1);
YTest = cell(taskNum, 1);
kfold = cell(foldNum, 1);
rng(seed);

for t=1:taskNum
    kfold{t} = crossvalind('KFold', Y{t}, foldNum);
end

avgTestAcc = zeros(taskNum, 1);
for fi=1:foldNum
    % get training and testing data of each task
    for t=1:taskNum
        XTrain{t} = X{t}(kfold{t} ~= fi, :);
        XTest{t} = X{t}(kfold{t} == fi, :);
        YTrain{t} = Y{t}(kfold{t} ~= fi, :);
        YTest{t} = Y{t}(kfold{t} == fi, :);
    end

    % do training and testing 
    [ W, c, YTrainPredict, trainAcc, YTestPredict, testAcc ] = TrainTest( ...
        XTrain, YTrain, XTest, YTest, method, p1, opts);
    avgTestAcc = avgTestAcc + testAcc;
end
avgTestAcc = avgTestAcc / foldNum;

end

