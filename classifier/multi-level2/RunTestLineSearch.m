function [ ] = RunTest( dataFile, method, update, stop, learn_rate, maxIter, seed, fid, debugLevel, outFilePrefix )
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
[c1Range, c2Range] = ParamRange( method, update);
params.learn_rate = learn_rate; 
params.maxIter = maxIter;
params.update = update;
params.tol = 1e-5;
params.stop = stop;
fprintf(2, 'method:%s learn_rate:%f maxIter:%d update:%s tol:%f stop:%s\n', method, learn_rate, maxIter, update, params.tol, stop);

%% training and testing 
rng(seed);
kfold = crossvalind('KFold', y, foldNum);
fprintf(2, 'In Seed:%d fold %d ... \n', seed, fid);
% get training and testing data of each task
[XTrain, yTrain, XTest, yTest] = split(X, y, kfold, fid);

% using best parameters to train classifier, and test it on testing
% data
for c1=c1Range
    c2=c1;
    %for c2=c2Range
    fprintf(2, 'c1:%f c2:%f\n', c1, c2);
    [ model, F_train, F_test, trainR, valR ] = trainLineSearch( method, ...
        XTrain, yTrain, c1, c2, params, 2, XTest, yTest );

    [ trainAcc, yTrainPredict ] = test( method, F_train, yTrain, model );
    [ testAcc, yTestPredict ] = test( method, F_test, yTest, model );
    fprintf('train: %f test:%f', trainAcc, testAcc);

    %end
end
end
