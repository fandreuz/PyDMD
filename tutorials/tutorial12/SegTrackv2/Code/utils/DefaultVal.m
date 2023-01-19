function tf = DefaultVal(in_var_names, in_var_defaults)
%% tf = isdefined(in_var_name)
%
% Returns a logical indicating if the variable “in_var_name” both exists and is
% not empty in the “caller” workspace.
%
% “in_var_name” is a text string that specifies the name of the variable to search for.
%
% Kent Conover, 12-Mar-08
% Fuxin Li modified from Kent Conover's code (online) on 1/17/2013
% Doesn't work to set default values as a vector

if ~iscell(in_var_names)
    in_var_names = {in_var_names};
end
if ~iscell(in_var_defaults)
    in_var_defaults = {in_var_defaults};
end
for i=1:length(in_var_names)
    cmd_txt = ['exist(''',in_var_names{i}, ''', ''var'');'];
    if evalin('caller', cmd_txt);
        cmd_txt = ['~isempty(',in_var_names{i}, ');'];
        if evalin('caller', cmd_txt);
            tf = true;
        else
            tf = false;
        end
    else
        tf = false;
    end
        if tf == false
            if isempty(in_var_defaults{i})
                in_var_defaults{i} = '[]';
            elseif isa(in_var_defaults{i}, 'numeric')
                in_var_defaults{i} = num2str(in_var_defaults{i});
            elseif isa(in_var_defaults{i},'char')
                in_var_defaults{i} = ['''' in_var_defaults{i} ''''];
            elseif isa(in_var_defaults{i},'logical')
                if in_var_defaults{i} == true
                    in_var_defaults{i} = 'true';
                else
                    in_var_defaults{i} = 'false';
                end
            end
            cmd_txt = [in_var_names{i} ' = ' in_var_defaults{i} ';'];
            evalin('caller', cmd_txt);
        end
end