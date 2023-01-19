function img_type = get_img_extension(img_dir, img_name)
    if nargin == 1
        files = dir([img_dir, '*']);
        img_name = img_dir;
    else
        files = dir(fullfile(img_dir, [img_name '*']));
        img_name = [img_dir img_name];
    end
    if isempty(files)
        error(['File ' img_name ' not found!']);
    end
    [name_len, id] = sort(arrayfun(@(a) numel(a.name), files), 'ascend');
    img_type = files(id(1)).name(end-3:end);
end