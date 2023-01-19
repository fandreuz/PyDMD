function merged = merge_img(segments, Is, cmap, strength)
    if ~iscell(segments)
        segments = {segments};
    end
    if ~iscell(Is)
        Is = {Is};
    end
    if ~exist('cmap','var') || isempty(cmap)
        cmap = [0 0 0;0 1 0];
    end
    if ~exist('strength','var') || isempty(strength)
        strength = 0.4;
    end
    for i=1:length(segments)
        rgbimg = uint8(ind2rgb(segments{i},cmap) * 255);
        merged{i} = immerge(Is{i}, rgbimg, double(segments{i} > 0) * strength);
    end
    if size(merged) == 1
        merged = merged{1};
    end
end