function [ W, c ] = Train( X, Y, method, p1, opts )
%TRAIN Summary of this function goes here
%   Detailed explanation goes here

c = zeros(1, length(Y));

if strcmp(method, 'Least_Lasso') == 1
    W = Least_Lasso(X, Y, p1, opts);
elseif strcmp(method, 'Logistic_Lasso') == 1
    [W, c] = Logistic_Lasso(X, Y, p1, opts);
elseif strcmp(method, 'Least_L21') == 1
    W = Least_L21(X, Y, p1, opts);
elseif strcmp(method, 'Logistic_L21') == 1
    [W, c] = Logistic_L21(X, Y, p1, opts);   
elseif strcmp(method, 'Least_Trace') == 1
    W = Least_Trace(X, Y, p1, opts);
elseif strcmp(method, 'Logistic_Trace') == 1
    [W, c] = Logistic_Trace(X, Y, p1, opts);      
end 

end

