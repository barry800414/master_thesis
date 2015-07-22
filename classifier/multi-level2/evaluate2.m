function [ r ] = evaluate2( method, F, y, model, c1, c2 )
%LEAST_MULTILEVEL_EVAL Summary of this function goes here
% calculate Loss function, prediction 
%   Detailed explanation goes here

nDocs = length(F);
err = 0.0;
yPredict = zeros(nDocs, 1);

if strcmp(method, 'Least') == 1
    for i=1:nDocs
        value = (model.w_subj' * F{i} * model.w_sta);
        yPredict(i) = sign(value);
        err = err + (value - y(i))^2;
    end
    err = err / nDocs;
    reg = c1 * (model.w_sta' * model.w_sta) + c2 * (model.w_subj' * model.w_subj);
    funcVal = err + reg;
elseif strcmp(method, 'Logistic') == 1
    for i=1:nDocs
        value = model.w_subj' * F{i} * model.w_sta + model.c;
        yPredict(i) = sign(value);
        err = err + log(1 + exp(-1*y(i)*(value)));
    end
    err = err / nDocs;
    reg = c1 * (model.w_sta' * model.w_sta) + c2 * (model.w_subj' * model.w_subj);
    funcVal = err + reg;
end

acc = nnz(y == yPredict) / nDocs;

r.func = funcVal;
r.err = err;
r.reg = reg;
r.acc = acc;
r.predict = yPredict;

end

