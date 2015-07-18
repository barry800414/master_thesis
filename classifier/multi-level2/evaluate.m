function [ funcVal, error, reg, acc, predict ] = evaluate( method, F, y, ...
    model, c1, c2 )
%LEAST_MULTILEVEL_EVAL Summary of this function goes here
% calculate Loss function, prediction 
%   Detailed explanation goes here

[funcVal, error, reg] = calcFuncVal( method, F, y, model, c1, c2 );
[acc, predict] = test( method, F, y, model );

end

