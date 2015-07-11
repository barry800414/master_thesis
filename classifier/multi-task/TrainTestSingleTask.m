function [ W, c, YTrainPredict,  trainAcc, YTestPredict, testAcc ] = TrainTestSingleTask( ...
XTrain, YTrain, XTest, YTest, method, p1, opts, t, taskProp)
%TRAINTEST Summary of this function goes here
% XTrain: a cell array of t n by d matrix
% 
%   Detailed explanation goes here

taskNum = length(YTrain);

% training 
[W, c] = Train(XTrain, YTrain, method, p1, opts);
 
if t <= taskNum
    % only test on single task
    % predict on training data
    [YTrainPredict, trainAcc] = Test(XTrain{t}, YTrain{t}, W(:, t), c(t), method);

    % predict on testing data
    [YTestPredict, testAcc] = Test(XTest{t}, YTest{t}, W(:, t), c(t), method);
else
    fprintf(2, 'Error');
end

end

