function quals = get_inds_quality(exp_dir, directories, img_names, mask_type, inds_mat, quality_type)
    quals = zeros(size(inds_mat))';
    for i=length(img_names):-1:1
        qual_base_dir = [exp_dir 'SegmentEval/' mask_type '/' quality_type '/' directories{i} '/'];
        var = load([qual_base_dir img_names{i} '.mat']);
        Quality = var.Quality;
        if length(Quality) > 1
                if ~iscell(quals)
                    quals = {quals};
                end
                for j=1:length(Quality)
                    nz = inds_mat(length(img_names)-i+1,:) ~= 0;
                    quals{j}(nz,i) = Quality(j).q(inds_mat(length(img_names)-i+1,nz))';
                end                
        else
            nz = inds_mat(length(img_names)-i+1,:) ~= 0;
            quals(nz,i) = Quality.q(inds_mat(length(img_names)-i+1,nz))';
        end
    end
end