function [ ] = RunTask( dataFile, method, update, stop, learn_rate, maxIter, seedNum, debugLevel, outFilePrefix )
% to do: 
% 1. gd using best iteration
% 2. logistic regression (done)
% 3. update strategy (rmsprop, adagrad
% 4. least bias term (done)
% 5. fix random seed (done) and multiple random seed?

%% load data
load(dataFile);
X.F_sta = F_sta;
X.F_subj = F_subj;
k1 = size(F_sta{1}); k1 = k1(2);
k2 = size(F_subj{1}); k2 = k2(2);

%% parameter settings
foldNum = 10;
[c1Range, c2Range] = ParamRange(method, update);
params.learn_rate = learn_rate; 
params.maxIter = maxIter;
params.update = update;
params.tol = 1e-5;
params.stop = stop;
fprintf(2, 'method:%s learn_rate:%f maxIter:%d update:%s tol:%f stop:%s\n', method, learn_rate, maxIter, update, params.tol, stop);

% open result file to write
fout = fopen(strcat(outFilePrefix, '_result.csv'), 'w');
fprintf(fout, 'framework, classifier, scorer, dimension, randSeed, foldNum, train, val, test\n');

%% training and testing 
for seed=1:seedNum
    rng(seed);
    kfold = crossvalind('KFold', y, foldNum);
    for fid=1:foldNum
        fprintf(2, 'In seed: %d fold %d ... \n', seed, fid);
        % get training and testing data of each task
        [XTrain, yTrain, XTest, yTest] = split(X, y, kfold, fid);

        % use cross validation to find best c1, c2
        [c1, c2, valAcc, bestIter ] = GridSearchCV ( method, XTrain, yTrain, c1Range, c2Range, ...
            params, foldNum, seed, debugLevel );
        fprintf(2, 'best c1:%f c2:%f iter:%d found by cross-validation\n', c1, c2, bestIter);

        % using best parameters to train classifier, and test it on testing data
        newParams = params; 
        newParams.stop = 'trainFunc'; 
        newParams.maxIter = bestIter;
        [ model, F_train, F_test ] = train( method, XTrain, yTrain, c1, c2, ...
            newParams, 2 );
        [ trainAcc, yTrainPredict ] = test( method, F_train, yTrain, model );
        [ testAcc, yTestPredict ] = test( method, F_test, yTest, model );
        fprintf(fout, 'MultiLevel, %s, Accuracy, %d %d, %d, %d, %f, %f, %f\n', ...
                method, k1, k2, seed, fid, trainAcc, valAcc, testAcc);
    end
end

fclose(fout);

