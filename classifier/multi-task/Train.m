function [ model ] = Train( X, Y, method, p, opts )
%TRAIN Summary of this function goes here
% p is a struct of paremeters (p1, p2, k ...)
% model is a struct of trained parameters (weight matrix, bias ...)
%   Detailed explanation goes here

c = zeros(1, length(Y));

if strcmp(method, 'Least_Lasso') == 1
    model.W = Least_Lasso(X, Y, p.p1, opts);
elseif strcmp(method, 'Logistic_Lasso') == 1
    [model.W, model.c] = Logistic_Lasso(X, Y, p.p1, opts);
elseif strcmp(method, 'Least_L21') == 1
    model.W = Least_L21(X, Y, p.p1, opts);
elseif strcmp(method, 'Logistic_L21') == 1
    [model.W, model.c] = Logistic_L21(X, Y, p.p1, opts);   
elseif strcmp(method, 'Least_Trace') == 1
    model.W = Least_Trace(X, Y, p.p1, opts);
elseif strcmp(method, 'Logistic_Trace') == 1
    [model.W, model.c] = Logistic_Trace(X, Y, p.p1, opts);      
elseif strcmp(method, 'Least_Dirty') == 1
    [model.W, funcVal, model.P, model.Q] = Least_Dirty(X, Y, p.p1, p.p2, opts);
elseif strcmp(method, 'Least_SparseTrace') == 1
    [model.W, funcVal, model.P, model.Q] = Least_SparseTrace(X, Y, p.p1, p.p2, opts);
elseif strcmp(method, 'Least_CASO') == 1
    [model.W, funcVal, model.M] = Least_CASO(X, Y, p.p1, p.p2, p.k, opts);
elseif strcmp(method, 'Logistic_CASO') == 1
    [model.W, funcVal, model.M] = Logistic_CASO(X, Y, p.p1, p.p2, p.k, opts);

end
end

