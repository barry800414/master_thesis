function [ acc, yPredict ] = test( method, F, y, model )
%TEST Summary of this function goes here
%   Detailed explanation goes here

nDocs = length(y);
yPredict = predict(method, F, model);
acc = nnz(y == yPredict) / nDocs;

end

