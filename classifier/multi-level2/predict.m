function [ yPredict ] = predict( method, F, model )
%PREDICT Summary of this function goes here
%   Detailed explanation goes here

nDocs = length(F);
yPredict = zeros(nDocs, 1);

if strcmp(method, 'Least') == 1
    for i=1:nDocs
        yPredict(i) = sign(model.w_subj' * F{i} * model.w_sta);
    end
elseif strcmp(method, 'Logistic') == 1
    for i=1:nDocs
        yPredict(i) = sign(model.w_subj' * F{i} * model.w_sta + model.c);
    end
end
end

