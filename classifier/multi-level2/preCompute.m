function [ F ] = preCompute( method, F_sta, F_subj )
%PRECOMPUTE Summary of this function goes here
%   Detailed explanation goes here
nDocs = length(F_sta);
F = cell(1, nDocs); 

if strcmp(method, 'Least') == 1
    for i=1:nDocs
        k1 = size(F_sta{i}); k1 = k1(1);
        sta = [F_sta{i}, ones(k1, 1)]; % append constant column
        k2 = size(F_subj{i}); k2 = k2(1);
        subj = [F_subj{i}, ones(k2, 1)]; % append constant column
        F{i} = subj' * sta; % (k2+1)x(k1+1) matrix
    end  
elseif strcmp(method, 'Logistic') == 1
    for i=1:nDocs
        F{i} = F_subj{i}' * F_sta{i}; % k2xk1 matrix
    end  
end
end

