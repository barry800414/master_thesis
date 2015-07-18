function [ output_args ] = print_eval( prefix, funcVal, error, reg, acc)
%PRINT_EVAL Summary of this function goes here
%   Detailed explanation goes here

fprintf(2, '%s:(%g/%g/%g/%.3f) ', prefix, funcVal, error, reg, acc);

end

