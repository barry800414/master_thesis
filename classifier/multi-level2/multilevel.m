
% input
% y: label of documents
% F_sta: length d cell array of Ni x k1 matrix (sentence stance feature
% matrix)
% F_subj: length d cell array of Ni x k2 matrix (sentence subjective
% feature matrix)
% c1: regularization term of w_sta
% c2: regularization term of w_subj

% k1: the dimension of stance feature
% k2: the dimension of subjective feature
k1 = size(F_sta{1}); k1 = k1(2);
k2 = size(F_subj{1}); k2 = k2(2);

% initilization 
nDocs = length(F_sta);
w_sta = rand(k1, 1); % k1 x 1 feature vector
w_subj = rand(k2, 1); % k2 x 1 feature vector
F = cell(nDocs); 
for i=1:nDocs
    F{i} = F_subj{i}' * F_sta{i}; % k2xk1 matrix
end    

% iteratively solve problem 
maxIter = 10;

for i=1:maxIter
    % fix w_subj, solve w_sta
    L = zeros(nDocs, k1);
    for j=1:nDocs
        L(:,j) = w_subj' * F{i}; 
    end
    A = (1.0/nDocs)*L'*L + c1 * eye(k1);
    b = L'*y;
    w_sta = A\b;
    
    % fix w_sta, solve w_subj
    R = zeros(nDocs, k2);
    for j=1:nDocs
        R(:,j) = w_sta' * F{i};
    end
    A = (1.0/nDocs)*R'*R + c2 * eye(k2);
    b = R'*y;
    w_subj = A\b;
    
    yPredict = predict(F, w_sta, w_subj);
    acc = nnz(y == yPredict) / nDocs;
    fprintf(stderr, 'Training accuracy: %f', acc);
end
