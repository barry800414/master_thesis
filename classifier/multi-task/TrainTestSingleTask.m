function [ model, YTrainPredict,  trainAcc, YTestPredict, testAcc ] = TrainTestSingleTask( ...
XTrain, YTrain, XTest, YTest, method, p, opts, t)
%TRAINTEST Summary of this function goes here
% XTrain: a cell array of t n by d matrix
% 
%   Detailed explanation goes here

taskNum = length(YTrain);

% training 
model = Train(XTrain, YTrain, method, p, opts);
 
if t <= taskNum
    % only test on single task
    % predict on training data
    [YTrainPredict, trainAcc] = Test(XTrain{t}, YTrain{t}, model.W(:, t), model.c(t), method);

    % predict on testing data
    [YTestPredict, testAcc] = Test(XTest{t}, YTest{t}, model.W(:, t), model.c(t), method);
else
    fprintf(2, 'Error');
end

end

