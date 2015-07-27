function [ avgTestAcc ] = OneTestKFoldTrainTest( X, Y, topic, foldNum, seed, method, p, opts )
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
rng(seed);
kfold = crossvalind('KFold', Y{topic}, foldNum);

avgTestAcc = zeros(taskNum, 1);
avgSparsity = 0.0;
for fi=1:foldNum
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
    % do training and testing
    [ model, YTrainPredict, trainAcc, YTestPredict, testAcc ] = TrainTest( ...
            XTrain, YTrain, XTest, YTest, method, p, opts);
    avgTestAcc = avgTestAcc + testAcc;
    wSize = size(model.W);
    avgSparsity = avgSparsity + (nnz(model.W) / (wSize(1) * wSize(2)));
end
avgTestAcc = avgTestAcc / foldNum;
avgSparsity = avgSparsity / foldNum;
fprintf(2, ' Sparsity: %.3f', avgSparsity);
end

