%% file example_mtl_classify.m
%   This example shows how to perform classification using least squares
%   loss and logistic loss. 
%
%% LICENSE
%   This program is free software: you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation, either version 3 of the License, or
%   (at your option) any later version.
%
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%
%   You should have received a copy of the GNU General Public License
%   along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
%   Copyright (C) 2011 - 2012 Jiayu Zhou and Jieping Ye 
%
%   You are suggested to first read the Manual.
%   For any problem, please contact with Jiayu Zhou via jiayu.zhou@asu.edu
%
%   Last modified on June 3, 2012.
%


clear, clc;
addpath('../MALSAR/utils/')
addpath('../MALSAR/functions/Lasso/')

n = 50; % number of samples (can be different for each task
d = 300; % number of dimension of feature vector
t = 10; %task number

X = cell(t, 1);
Y = cell(t, 1);
W = randn(d, t); % d * t matrix, each column is weight vector for that task
W_mask = abs(randn(d, t))<1;
W(W_mask) = 0;
for i = 1: t
    X{i} = randn(n, d);  % each x is an n * d matrix 
    Y{i} = sign(X{i} * W(:, i) + rand(n, 1) * 0.01);  % each y is a n * 1 vector
    size(Y{i})
end


% training and prediction using least squares loss
X
Y
W_pred = Least_Lasso(X, Y, 0.01);  % the weight by training 

% compute training error
least_acc = zeros(t, 1);  
for i = 1: t
    least_acc(i) = nnz(sign(X{i} * W_pred(:, i)) == Y{i})/n;
end
fprintf('Least Squares Loss Training Accuracy: %.4f +/- %.4f\n', mean(least_acc), std(least_acc));


% training and prediction using logistic loss
[W_pred C_pred]= Logistic_Lasso(X, Y, 0.01);
% compute training error
logistic_acc = zeros(t, 1);
for i = 1: t
    logistic_acc(i) = nnz(sign(X{i} * W_pred(:, i) + C_pred(i)) == Y{i})/n;
end
fprintf('Logistic Loss Training Accuracy: %.4f +/- %.4f\n', mean(logistic_acc), std(logistic_acc));





