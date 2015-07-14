function [ model, YTrainPredict, trainAcc, YTestPredict, testAcc ] = TrainTest( ...
XTrain, YTrain, XTest, YTest, method, p, opts)
%TRAINTEST Summary of this function goes here
% XTrain: a cell array of t n by d matrix
% 
%   Detailed explanation goes here

taskNum = length(YTrain);

% training 
model = Train(XTrain, YTrain, method, p, opts);
    
YTrainPredict = cell(taskNum, 1);
YTestPredict = cell(taskNum, 1);
trainAcc = zeros(taskNum, 1);
testAcc = zeros(taskNum, 1);

% predict on training data
for i = 1: taskNum
    [YTrainPredict{i}, trainAcc(i)] = Test(XTrain{i}, YTrain{i}, model.W(:, i), model.c(i), method);
end

% predict on testing data
for i = 1:taskNum
    [YTestPredict{i}, testAcc(i)] = Test(XTest{i}, YTest{i}, model.W(:, i), model.c(i), method);
end

end

