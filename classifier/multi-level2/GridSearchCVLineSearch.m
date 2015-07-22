function [ bestC1, bestC2, avgValAcc ] = GridSearchCVLineSearch ( method, X, y, c1Range, ...
    c2Range, params, foldNum, seed, debugLevel )
%GRIDSEARCHCV Summary of this function goes here
%   Detailed explanation goes here

% cross-validation to find best parameters
rng(seed);
kfold = crossvalind('KFold', y, foldNum);
bestAcc = 0.0;
bestC1 = 0.0;
bestC2 = 0.0;
bestIter = 0;
for c1=c1Range
    %for c2=c2Range
    c2 = c1;
    fprintf(2, 'Search at c1:%f c2:%f ...\n', c1, c2);
    avgTrainAcc = 0.0;
    avgValAcc = 0.0;
    avgIter = 0;
    for fid=1:foldNum
        % get training and testing data of each task
        [XTrain, yTrain, XVal, yVal] = split(X, y, kfold, fid);
        
        % training 
        [ model, F_train, F_val, iter, trainR, valR ] = trainLineSearch( ...
            method, XTrain, yTrain, c1, c2, params, debugLevel, XVal, yVal );

        % testing
        avgTrainAcc = avgTrainAcc + trainR.acc;
        avgValAcc = avgValAcc + valR.acc;
        avgIter = avgIter + iter;
        fprintf(2, 'Iter:%d trainAcc:%f valAcc:%f\n', iter, trainR.acc, valR.acc);
    end
    avgTrainAcc = avgTrainAcc / foldNum;
    avgValAcc = avgValAcc / foldNum;
    avgIter = int32(avgIter / foldNum);
    fprintf(2, 'avgIter:%d avgTrainAcc:%f avgValAcc:%f\n', avgIter, avgTrainAcc, avgValAcc); 

    if avgValAcc > bestAcc
        bestAcc = avgValAcc;
        bestC1 = c1;
        bestC2 = c2;
    end
    %end
end


end

