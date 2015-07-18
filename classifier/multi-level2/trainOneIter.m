function [ m ] = trainOneIter( method, F, y, m, c1, c2, learn_rate, update )
%TRAINONEITER Summary of this function goes here
%   Detailed explanation goes here
% m: model

[k2, k1] = size(F{1});
nDocs = length(y);

% least square problem updating 
if strcmp(method, 'Least') == 1
    % fix m.w_subj, solve m.w_sta
    L = zeros(nDocs, k1);
    for j=1:nDocs
        L(j,:) = m.w_subj' * F{j};
    end
    if strcmp(update, 'ALS') == 1
        A = (1.0/nDocs)*(L'*L) + c1 * eye(k1);
        b = L'*y;
        m.w_sta = A\b;
        %m.w_sta = inv(A) *b;
    elseif strcmp(update, 'GD') == 1
        g = (1.0/nDocs) * (L'*L*m.w_sta - L'*y) + c1 * m.w_sta;
        m.w_sta = m.w_sta - learn_rate * g;
    end

    % fix m.w_sta, solve m.w_subj
    R = zeros(nDocs, k2);
    for j=1:nDocs
        R(j,:) = m.w_sta' * F{j}';
    end
    if strcmp(update, 'ALS') == 1
        A = (1.0/nDocs)*(R'*R) + c2 * eye(k2);
        b = R'*y;
        m.w_subj = A\b;
        %m.w_subj = inv(A) *b;
    elseif strcmp(update, 'GD') == 1
        g = (1.0/nDocs) * (R'*R*m.w_subj - R'*y) + c2 * m.w_subj;
        m.w_subj = m.w_subj - learn_rate * g;
    end
% logistic problem updating 
elseif strcmp(method, 'Logistic') == 1
    % fix m.w_subj, solve m.w_sta
    g = 0.0;
    for i=1:nDocs
        L = m.w_subj' * F{i};
        g = g + ((-1)*y(i)*L') / (1 + exp(y(i)*((L * m.w_sta) + m.c)));
    end
    g = g/nDocs + 2 * c1 * m.w_sta;
    m.w_sta = m.w_sta - learn_rate * g;

    % fix m.w_sta, solve m.w_subj
    g = 0.0;
    for i=1:nDocs
        R = m.w_sta' * F{i}';
        g = g + ((-1)*y(i)*R') / (1 + exp(y(i)*((R * m.w_subj) + m.c)));
    end
    g = g/nDocs + 2 * c2 * m.w_subj;
    m.w_subj = m.w_subj - learn_rate * g;
end
end



