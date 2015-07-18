function [ bestC1, bestC2, bestAcc, bestIter ] = GridSearchCV ( method, X, y, c1Range, ...
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
    avgBestIter = 0;
    for fid=1:foldNum
        % get training and testing data of each task
        [XTrain, yTrain, XVal, yVal] = split(X, y, kfold, fid);
        
        % training 
        [ model, F_train, F_val, iter, trainAcc, valAcc ] = train( ...
            method, XTrain, yTrain, c1, c2, params, debugLevel, XVal, yVal );

        % testing
        %[ trainAcc ] = test( method, F_train, yTrain, model );
        %[ valAcc ] = test( method, F_val, yVal, model );
        avgTrainAcc = avgTrainAcc + trainAcc;
        avgValAcc = avgValAcc + valAcc;
        avgBestIter = avgBestIter + iter;
        fprintf(2, 'bestIter:%d trainAcc:%f valAcc:%f\n', iter, trainAcc, valAcc);
    end
    avgTrainAcc = avgTrainAcc / foldNum;
    avgValAcc = avgValAcc / foldNum;
    avgBestIter = int32(avgBestIter / foldNum);
    fprintf(2, 'avgTrain:%f avgVal:%f\n', avgTrainAcc, avgValAcc); 
    if avgValAcc > bestAcc
        bestAcc = avgValAcc;
        bestC1 = c1;
        bestC2 = c2;
        bestIter = iter;
    end
    %end
end


end

