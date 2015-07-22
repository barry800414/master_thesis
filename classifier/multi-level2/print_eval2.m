function [ output_args ] = print_eval( prefix, r )
%PRINT_EVAL Summary of this function goes here
%   Detailed explanation goes here

fprintf(2, '%s:(%g/%g/%g/%.3f) ', prefix, r.func, r.err, r.reg, r.acc);

end

