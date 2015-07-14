function [ bestP, bestAcc ] = GridSearchCV( X, Y, foldNum, seed, method, pRange, opts )
%TRAINVALTEST Summary of this function goes here
% bestP1: the best parameter p1 by grid search 
% bestWeightedAcc: the best weighted average at that parameter
% bestAcc: the best average of each task at that parameter (taskNum by 1 vector)
%   Detailed explanation goes here

% generate 10-fold training and testing data
taskNum = length(Y);
taskProp = zeros(1, taskNum); % proportion of data of each task

for t=1:taskNum
    taskProp(t) = length(Y{t});
end
taskProp = taskProp / sum(taskProp);

% 10-fold testing 
bestP = struct;
bestWeightedAcc = -1.0;
bestAcc = zeros(taskNum, 1);

oneParamMethod = { 'Least_Lasso', 'Logistic_Lasso', 'Least_L21', 'Logistic_L21', 'Least_Trace', 'Logistic_Trace' };
twoParamMethod = { 'Least_Dirty', 'Least_SparseTrace' };
threeParamMethod = { 'Least_CASO', 'Logistic_CASO' };

if ismember(method, oneParamMethod)
    for p1 = pRange.p1Range
        fprintf(2, 'search at p1:%f ', p1);
        p.p1 = p1;
        avgTestAcc = KFoldTrainTest(X, Y, foldNum, seed, method, p, opts);
        for t=1:taskNum
            fprintf(2, ' t%d: %.3f', t, avgTestAcc(t));
        end
        fprintf(2, '\n');
        weightedAcc = taskProp * avgTestAcc;
        if weightedAcc > bestWeightedAcc
            bestP = p;
            bestWeightedAcc = weightedAcc;
            bestAcc = avgTestAcc;
        end
    end
elseif ismember(method, twoParamMethod)
    for p1 = pRange.p1Range
        for p2 = pRange.p2Range
            fprintf(2, 'search at p1:%f p2:%f', p1, p2);
            p.p1 = p1; p.p2 = p2;
            avgTestAcc = KFoldTrainTest(X, Y, foldNum, seed, method, p, opts);
            for t=1:taskNum
                fprintf(2, ' t%d: %.3f', t, avgTestAcc(t));
            end
            fprintf(2, '\n');
            weightedAcc = taskProp * avgTestAcc;
            if weightedAcc > bestWeightedAcc
                bestP = p;
                bestWeightedAcc = weightedAcc;
                bestAcc = avgTestAcc;
            end
        end
    end
elseif ismember(method, threeParamMethod)
    for p1 = pRange.p1Range
        for p2 = pRange.p2Range
            for k = pRange.kRange
                fprintf(2, 'search at p1:%f p2:%f k:%d', p1, p2, k);    
                p.p1 = p1; p.p2 = p2; p.k = k;
                avgTestAcc = KFoldTrainTest(X, Y, foldNum, seed, method, p, opts);
                for t=1:taskNum
                    fprintf(2, ' t%d: %.3f', t, avgTestAcc(t));
                end
                fprintf(2, '\n');
                weightedAcc = taskProp * avgTestAcc ;
                if weightedAcc > bestWeightedAcc
                    bestP = p;
                    bestWeightedAcc = weightedAcc;
                    bestAcc = avgTestAcc;
                end
            end
        end
    end
else
    fprintf(2, 'Unknown method\n');
end

end
