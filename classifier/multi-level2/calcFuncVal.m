function [ funcVal, err, reg ] = calcFuncVal( method, F, y, model, c1, c2 )
%PREDICT Summary of this function goes here
% calculate Loss function value
%   Detailed explanation goes here

nDocs = length(F);
err = 0.0;

if strcmp(method, 'Least') == 1
    for i=1:nDocs
        err = err + ((model.w_subj' * F{i} * model.w_sta) - y(i))^2;
    end
    err = err / nDocs;
    reg = c1 * (model.w_sta' * model.w_sta) + c2 * (model.w_subj' * model.w_subj);
    funcVal = err + reg;
elseif strcmp(method, 'Logistic') == 1
    for i=1:nDocs
        err = err + log(1 + exp(-1*y(i)*(model.w_subj' * F{i} * model.w_sta + model.c)));
    end
    err = err / nDocs;
    reg = c1 * (model.w_sta' * model.w_sta) + c2 * (model.w_subj' * model.w_subj);
    funcVal = err + reg;
end

end

