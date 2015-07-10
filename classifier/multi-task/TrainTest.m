function [ W, c, YTrainPredict,  trainAcc, YTestPredict, testAcc ] = TrainTest( ...
XTrain, YTrain, XTest, YTest, method, p1, opts)
%TRAINTEST Summary of this function goes here
% XTrain: a cell array of t n by d matrix
% 
%   Detailed explanation goes here

taskNum = length(YTrain);

% training 
[W, c] = Train(XTrain, YTrain, method, p1, opts);
    
YTrainPredict = cell(taskNum, 1);
YTestPredict = cell(taskNum, 1);
trainAcc = zeros(taskNum, 1);
testAcc = zeros(taskNum, 1);

% predict on training data
for i = 1: taskNum
    [YTrainPredict{i}, trainAcc(i)] = Test(XTrain{i}, YTrain{i}, W(:, i), c(i), method);
end

% predict on testing data
for i = 1:taskNum
    [YTestPredict{i}, testAcc(i)] = Test(XTest{i}, YTest{i}, W(:, i), c(i), method);
end

end

