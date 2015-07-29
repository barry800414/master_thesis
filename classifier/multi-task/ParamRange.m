function [ pRange ] = ParamRange( method )
%TRAINVALTEST Summary of this function goes here
%   Detailed explanation goes here

oneParamMethod = { 'Least_Lasso', 'Logistic_Lasso', 'Least_L21', 'Logistic_L21', 'Least_Trace', 'Logistic_Trace' };
twoParamMethod = { 'Least_Dirty', 'Least_SparseTrace' };
threeParamMethod = { 'Least_CASO', 'Logistic_CASO' };

if strcmp(method, 'Least_Lasso') == 1
    pRange.p1Range = 2.^(-9:1:8);
elseif strcmp(method, 'Logistic_Lasso') == 1
    %pRange.p1Range = 2.^(-15:2:-1);
    pRange.p1Range = 2.^(-13:2:-3);
elseif strcmp(method, 'Least_L21') == 1
    pRange.p1Range = 2.^(-2:1:8);
elseif strcmp(method, 'Logistic_L21') == 1
    %pRange.p1Range = 2.^(-13:2:1);
    pRange.p1Range = 2.^(-11:2:-1);
elseif strcmp(method, 'Least_Trace') == 1
    pRange.p1Range = 2.^(-7:1:8);
elseif strcmp(method, 'Logistic_Trace') == 1
    %pRange.p1Range = 2.^(-5:1:3);
    pRange.p1Range = [0.01:0.01:0.1 0.2:0.1:1 2:1:10];
elseif strcmp(method, 'Least_Dirty') == 1
    pRange.p1Range = 2.^(2:2:6);
    pRange.p2Range = 2.^(-4:2:6);
elseif strcmp(method, 'Least_SparseTrace') == 1
    pRange.p1Range = 2.^(-2:1:6);
    pRange.p2Range = 2.^(-8:1:4);
elseif ismember(method, threeParamMethod)
    pRange.p1Range = 10.^(-3:1:3);
    pRange.p2Range = 10.^(-3:1:3);
    pRange.kRange = [20, 30, 40, 50];
else
    fprintf(2, 'Unknown method\n');
end

end
