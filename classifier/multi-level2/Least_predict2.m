function [ yPredict ] = predict2( F_sta, F_subj, w_sta, w_subj )
%PREDICT Summary of this function goes here
%   Detailed explanation goes here

nDocs = length(F_subj);
yPredict = zeros(nDocs, 1);
for i=1:nDocs
    (w_subj' * F_subj{i}')'
    %aaaa =(w_subj' * F_subj{i}')';
    %F_sta{i} * w_sta
    yPredict(i) = sign(w_subj' * F_subj{i}' * F_sta{i} * w_sta);
end

end

