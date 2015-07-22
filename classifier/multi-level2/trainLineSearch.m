function [ model, F_train, F_val, nowIter, trainR, valR ] = trainLineSearch( method, XTrain, ...
    yTrain, c1, c2, params, debugLevel, XVal, yVal )
%TRAIN Summary of this function goes here
%   Detailed explanation goes here

% XTrain: a struct containing F_sta and F_subj
% yTrain: label of training documents 
% F_sta: length d cell array of Ni x k1 matrix (sentence stance feature
% matrix)
% F_subj: length d cell array of Ni x k2 matrix (sentence subjective
% feature matrix)
% c1: regularization term of w_sta
% c2: regularization term of w_subj
% params: other parameters of model (learn_rate, tol, maxIter, update ...)
    % learn_rate: learning rate
    % tol: threshold of termination
    % update: update method
    % stop: stopping criteria (trainFunc, valFunc, valAcc, valErr(without reg term) )
    % maxIter: maximum iteration
% XVal, yVal: if given, this is the validation data

%% get input
F_sta_train = XTrain.F_sta;
F_subj_train = XTrain.F_subj;
if isfield(params, 'learn_rate') learn_rate = params.learn_rate; else learn_rate = 0.001; end
if isfield(params, 'tol') tol = params.tol; else tol = 0.00001; end
if isfield(params, 'update') update = params.update; else update = 'GD'; end 
if isfield(params, 'stop') stop = params.stop; else stop = 'valFunc'; end
if isfield(params, 'scale') scale = params.scale; else scale = 1.2; end
if isfield(params, 'maxIter') 
    maxIter = params.maxIter; 
else
    if strcmp(update,'GD') == 1 
        maxIter = 1000;
    elseif strcmp(update, 'ALS') == 1
        maxIter = 10;
    end
end
F_sta_val = XVal.F_sta;
F_subj_val = XVal.F_subj;

%% pre-compute 
F_train = preCompute( method, F_sta_train, F_subj_train );
F_val = preCompute( method, F_sta_val, F_subj_val );

%% initialize the model
[k2, k1] = size(F_train{1});
nDocs = length(F_train);
if isfield(params, 'w_sta') model.w_sta = params.w_sta; else model.w_sta = rand(k1, 1)/nDocs; end 
if isfield(params, 'w_subj') model.w_subj = params.w_subj; else model.w_subj = rand(k2, 1)/nDocs; end
if isfield(params, 'c') model.c = params.c; else model.c = rand(1); end

%% iteratively solve problem 
now_rate = learn_rate;
valR.func = inf;
valR.err = inf;
valR.reg = inf;
valR.acc = -inf;
for nowIter=1:maxIter
    % train one iteration by line search
    [ model, now_rate, toStop, trainR, valR ] = trainOneIterLineSearch( scale, method, ...
        F_train, yTrain, F_val, yVal, model, c1, c2, now_rate, update, valR );
    % print training function value, error, regularization term and accuracy
    if debugLevel >= 2
        fprintf(2, 'Iter %d:', nowIter);
        print_eval2('Train', trainR );
        print_eval2('Val', valR );
        fprintf(2, 'learn_rate: %f\n', now_rate);
    end
    if toStop
        break
    end
end

end

