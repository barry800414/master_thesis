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
%c1Range = 2.^(-15:2:7);
params.learn_rate = learn_rate; 
params.maxIter = maxIter;
params.update = update;
params.tol = 1e-5;
params.stop = stop;
params.scale = 1.1;
%k1 = size(F_sta{1}); k1=k1(2)+1;
%k2 = size(F_subj{1}); k2=k2(2)+1;

%params.w_sta = rand(k1, 1)/(k1*k2);
%params.w_subj = rand(k2, 1)/(k1*k2);

fprintf(2, 'method:%s learn_rate:%f maxIter:%d update:%s tol:%f stop:%s\n', method, learn_rate, maxIter, update, params.tol, stop);

%% training and testing 
rng(seed);
kfold = crossvalind('KFold', y, foldNum);
fprintf(2, 'In Seed:%d fold %d ... \n', seed, fid);
% get training and testing data of each task
[XTrain, yTrain, XTest, yTest] = split(X, y, kfold, fid);

% using best parameters to train classifier, and test it on testing data

predict = zeros(length(yTest), 1);
for c1=c1Range
    c2=c1;
    fprintf(2, 'c1:%f c2:%f\n', c1, c2);
    %[ model, F_train, F_test, iter, trainR, testR ] = trainLineSearch( method, XTrain, yTrain, c1, c2, ...
    %        params, 1, XTest, yTest );
    [ model, F_train, F_test, iter, trainAcc, testAcc ] = train( method, XTrain, yTrain, c1, c2, ...
            params, debugLevel, XTest, yTest );

    %fprintf(2, 'iter:%d trainAcc:%f testAcc:%f\n', iter, trainR.acc, testR.acc);
    fprintf(2, 'iter:%d trainAcc:%f testAcc:%f\n', iter, trainAcc, testAcc);

end
end
