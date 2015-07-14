function [ ] = PrintParam( param )
%TRAINVALTEST Summary of this function goes here
%   Detailed explanation goes here

if iscell(param)
    for t=1:length(param)
        p = param{t};
        fields = fieldnames(p);
        fprintf(2, 'Task %d:', t);
        for i = 1:numel(fields)
            fprintf(2, ' %s:%f', fields{i}, p.(fields{i}));
        end
        fprintf(2, '\n');
    end
else
    fields = fieldnames(param);
    for i = 1:numel(fields)
        fprintf(2, ' %s:%f', fields{i}, param.(fields{i}));
    end
    fprintf(2, '\n');
end
end
