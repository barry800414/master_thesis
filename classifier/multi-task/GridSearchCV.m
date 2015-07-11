function [ bestP1, bestAcc ] = GridSearchCV( X, Y, foldNum, seed, method, p1Range, opts )
%TRAINVALTEST Summary of this function goes here
% bestP1: the best parameter p1 by grid search 
% bestWeightedAcc: the best weighted average at that parameter
% bestAcc: the best average of each task at that parameter (taskNum by 1 vector)
%   Detailed explanation goes here

% generate 10-fold training and testing data
taskNum = length(Y);
taskProp = zeros(taskNum, 1); % proportion of data of each task

for t=1:taskNum
    taskProp(t) = length(Y{t});
end
taskProp = taskProp / sum(taskProp);

% 10-fold testing 
bestP1 = 0.0;
bestAcc = zeros(taskNum, 1);
bestWeightedAcc = -1.0;
for p1 = p1Range
    fprintf(2, 'search at p1:%f\n', p1);
    avgTestAcc = KFoldTrainTest(X, Y, foldNum, seed, method, p1, opts);
    % calcualte weighted average as validation score
    weightedAcc = avgTestAcc' * taskProp;
    if weightedAcc > bestWeightedAcc
        bestP1 = p1;
        bestWeightedAcc = weightedAcc;
        bestAcc = avgTestAcc;  
    end
end

