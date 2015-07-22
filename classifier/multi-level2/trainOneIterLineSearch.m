function [ m, new_rate, toStop, trainR, newValR ] = trainOneIterLineSearch( scale, method, F, y, ...
    F_val, yVal, m, c1, c2, learn_rate, update, valR )
%TRAINONEITER Summary of this function goes here
%   Detailed explanation goes here
% m: model

[k2, k1] = size(F{1});
nDocs = length(y);
new_rate = learn_rate;

maxTrial = 20;
trial = 0;
toStop = false;

% least square problem updating 
if strcmp(method, 'Least') == 1
    L = zeros(nDocs, k1);
    for j=1:nDocs
        L(j,:) = m.w_subj' * F{j};
    end
    R = zeros(nDocs, k2);
    for j=1:nDocs
        R(j,:) = m.w_sta' * F{j}';
    end

    % line search
    while true
        % fix m.w_subj, solve m.w_sta
        g = (1.0/nDocs) * (L'*L*m.w_sta - L'*y) + c1 * m.w_sta;
        newM.w_sta = m.w_sta - new_rate * g;

        % fix m.w_sta, solve m.w_subj
        g = (1.0/nDocs) * (R'*R*m.w_subj - R'*y) + c2 * m.w_subj;
        newM.w_subj = m.w_subj - new_rate * g;
        
        newValR = evaluate2( method, F_val, yVal, newM, c1, c2 );

        if newValR.func < valR.func
            m = newM;
            new_rate = new_rate * scale;
            break
        else
            if trial >= maxTrial
                toStop = true;
                newValR = valR;
                break;
            else
                new_rate = new_rate / scale;
                trial = trial + 1;
            end
        end
    end
elseif strcmp(method, 'Logistic') == 1
    L = zeros(nDocs, k1);
    for j=1:nDocs
        L(j,:) = m.w_subj' * F{j};
    end
    R = zeros(nDocs, k2);
    for j=1:nDocs
        R(j,:) = m.w_sta' * F{j}';
    end

    % line search
    while true
        g = 0.0;
        cg = 0.0;
        for i=1:nDocs
            value = (-1)*y(i) / (1 + exp(y(i)*((L(i,:) * m.w_sta) + m.c))); % gradient of c
            cg = cg + value;
            g = g + L(i,:)' * value;
        end
        cg = cg/nDocs;
        g = g/nDocs + 2 * c1 * m.w_sta;
        newM.w_sta = m.w_sta - new_rate * g;
        newM.c = m.c - new_rate * cg;
        
        % fix m.w_sta, solve m.w_subj
        g = 0.0;
        for i=1:nDocs
            g = g + ((-1)*y(i)*R(i,:)') / (1 + exp(y(i)*((R(i,:) * m.w_subj) + m.c)));
        end
        g = g/nDocs + 2 * c2 * m.w_subj;
        newM.w_subj = m.w_subj - new_rate * g;

        newValR = evaluate2( method, F_val, yVal, newM, c1, c2 );

        if newValR.func < valR.func
            m = newM;
            new_rate = new_rate * scale;
            break
        else
            if trial >= maxTrial
                toStop = true;
                newValR = valR;
                break;
            else
                new_rate = new_rate / scale;
                trial = trial + 1;
            end
        end
    end

end
trainR = evaluate2( method, F, y, m, c1, c2 );

