function [ predict, acc ] = Test( X, y, w, c, method )
%TEST Summary of this function goes here
% here X, y, W, c are only for one task
%   Detailed explanation goes here

if length(X) == 0
    predict = [];
    acc = 0.0;
    return;
end

if strcmp(method, 'Least_Lasso') == 1
    predict = sign(X * w);
elseif strcmp(method, 'Logistic_Lasso') == 1
    predict = sign(X * w + c);
elseif strcmp(method, 'Least_L21') == 1
    predict = sign(X * w);
elseif strcmp(method, 'Logistic_L21') == 1
    predict = sign(X * w + c);  
elseif strcmp(method, 'Least_Trace') == 1
    predict = sign(X * w);
elseif strcmp(method, 'Logistic_Trace') == 1
    predict = sign(X * w + c);     
elseif strcmp(method, 'Least_Dirty') == 1
    predict = sign(X * w);
elseif strcmp(method, 'Least_SparseTrace') == 1
    predict = sign(X * w);
elseif strcmp(method, 'Least_CASO') == 1
    predict = sign(X * w);
elseif strcmp(method, 'Logistic_CASO') == 1
    predict = sign(X * w + c);  
end 
acc = nnz(predict == y)/length(y);

end

