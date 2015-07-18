function [ XTrain, yTrain, XTest, yTest ] = split( X, y, kfold, fid )
%SPLIT Summary of this function goes here
%   Detailed explanation goes here

F_sta = X.F_sta;
F_subj = X.F_subj;


nTrain = nnz(kfold ~= fid);
nTest = nnz(kfold == fid);
F_sta_train = cell(1, nTrain);
F_sta_test = cell(1, nTest);
F_subj_train = cell(1, nTrain);
F_subj_test = cell(1, nTest);
trainIndex = 1;
testIndex = 1;
for i=1:length(kfold)
    foldId = kfold(i);
    if foldId ~= fid  % training data
        F_sta_train{trainIndex} = F_sta{i};
        F_subj_train{trainIndex} = F_subj{i};
        trainIndex = trainIndex + 1;
    else % testing data
        F_sta_test{testIndex} = F_sta{i};
        F_subj_test{testIndex} = F_subj{i};
        testIndex = testIndex + 1;
    end
end

XTrain.F_sta = F_sta_train;
XTrain.F_subj = F_subj_train;
XTest.F_sta = F_sta_test;
XTest.F_subj = F_subj_test;

yTrain = y(kfold ~= fid, :);
yTest = y(kfold == fid, :);
        
end



