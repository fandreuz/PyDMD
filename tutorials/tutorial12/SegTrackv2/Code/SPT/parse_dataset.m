function [directories, img_names] = parse_dataset(imgsetpath, dataset)
% Parse the dataset file
    DefaultVal('dataset',[]);
    img_names = textread(sprintf(imgsetpath, dataset),'%s');
    directories = cell(length(img_names),1);
    cur_folder = '';
    i = 1;
    while i <= length(img_names)
        % Meta dataset file linking to other data files
        if img_names{i}(1) == '*'
            try
                tmp = textread(sprintf(imgsetpath, img_names{i}(2:end)), '%s');
            catch E
                error(['Failed to read dataset file ' img_names{i}(2:end)]);
            end
            img_names = [img_names(1:i-1);img_names(i+1:end);tmp];
            directories = [directories(1:i-1);directories(i+1:end);cell(length(tmp),1)];
            continue;
        end
        % Folder
        if img_names{i}(end) == '/' || img_names{i}(end) == '\'
            cur_folder = img_names{i}(1:end-1);
            img_names = [img_names(1:i-1);img_names(i+1:end)];
            directories = [directories(1:i-1);directories(i+1:end)];
            continue;
        end
        directories{i} = cur_folder;
        i = i+1;
    end
end