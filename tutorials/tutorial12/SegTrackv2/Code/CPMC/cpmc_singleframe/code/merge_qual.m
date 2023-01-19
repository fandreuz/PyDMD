function [merged_qualities, max_merge, sz] = merge_qual(quality_struct, starter)
    labels = [quality_struct.class];
    quality = {quality_struct.q};
% Make sure it's a column vector
    if size(quality{1},1) < size(quality{1},2)
        quality = cellfun(@transpose,quality,'UniformOutput',false);
    end
    label_max = max(labels);
    fin_s = [starter length(labels)+1];

    merged_qualities = cell(length(starter), label_max);
    max_merge = cell(length(starter),1);
    sz = zeros(length(starter),1);
    for i=1:length(starter)
        % Empty image, just continue
        if starter(i)==0
            continue;
        end
        merge_start = fin_s(i);
        merge_end = fin_s(i+1) -1;
        [max_merge{i},pos1] = max([quality{merge_start:merge_end}],[],2);
        if isempty(max_merge{i})
            continue;
        end
        if isfield(quality_struct, 'sz')
            [val, pos] = max(max_merge{i});
            ps = find(pos + merge_start - 1 < fin_s, 1, 'first') - 1;
            sz(i) = quality_struct(pos1(pos) + merge_start - 1).sz(pos);
        end
        for j = merge_start:merge_end
            if isempty(merged_qualities{i, labels(j)})
                merged_qualities{i,labels(j)} = quality{j};
            else
                merged_qualities{i,labels(j)} = max(merged_qualities{i,labels(j)}, quality{j});
            end
        end
    end
end
