function [ bestValue, bestIter, bestTrainAcc, bestValAcc ] = keep_best_iter( ...
    method, bestValue, bestIter, bestTrainAcc, bestValAcc, nowIter, trainAcc, ...
    valFunc, valErr, valAcc )
%LEAST_MULTILEVEL_EVAL Summary of this function goes here
% calculate Loss function, prediction 
%   Detailed explanation goes here

if strcmp(method, 'trainFunc') == 1
    bestValue = -1.0;
    bestIter = nowIter;
    bestTrainAcc = trainAcc;
    bestValAcc = valAcc;
elseif strcmp(method, 'valFunc') == 1
    if bestValue == -1.0 || (valFunc < bestValue)
        bestValue = valFunc;
        bestIter = nowIter;
        bestTrainAcc = trainAcc;
        bestValAcc = valAcc;
    end
elseif strcmp(method, 'valErr') == 1
    if bestValue == -1.0 || (valErr < bestValue)
        bestValue = valErr;
        bestIter = nowIter;
        bestTrainAcc = trainAcc;
        bestValAcc = valAcc;
    end
elseif strcmp(method, 'valAcc') == 1
    if bestValue == -1.0 || (valAcc > bestValue)
        bestValue = valAcc;
        bestIter = nowIter;
        bestTrainAcc = trainAcc;
        bestValAcc = valAcc;
    end
end


end

