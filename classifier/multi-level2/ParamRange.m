function [ c1Range, c2Range ] = ParamRange( method, update )
%LEAST_MULTILEVEL_EVAL Summary of this function goes here
% calculate Loss function, prediction 
%   Detailed explanation goes here

if strcmp(update, 'ALS')
    c1Range = 2.^(-12:1:10);
    c2Range = 2.^(-12:1:10);
elseif strcmp(update, 'GD')
    c1Range = 2.^(-12:2:0);
    c2Range = 2.^(-12:2:0);
end
end

