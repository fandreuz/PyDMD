function fin_overl = SegTrack_find_best_qual(ConfigFile_or_opts,  dataset,quality_type, mask_type, minima)
% Test a segment ranker on all the images in the dataset data_test
% Rank segments using trees from simple segment features
% If not supplied mask_type_test, then it is taken from SVMSEGMopts
    if ~isstruct(ConfigFile_or_opts)
        eval(ConfigFile_or_opts);
    else
        SVMSEGMopts = ConfigFile_or_opts;
    end
    DefaultVal('mask_type', SVMSEGMopts.mask_type);
    [directories, img_names] = parse_dataset(SVMSEGMopts.seg.imgsetpath, dataset);
    all_overl = [];
    cov_obj = logical(0);
    for i=1:length(img_names)
        var = load([SVMSEGMopts.segment_quality_dir mask_type '/' quality_type '/' directories{i} '/' img_names{i} '.mat'],'Quality');
        if exist('minima','var') && ~isempty(minima) && minima
            all_overl(i,1:numel(var.Quality)) = min([var.Quality.q],[],1);
        else
            all_overl(i,1:numel(var.Quality)) = max([var.Quality.q],[],1);
        end
        cov_obj(i,1:numel(var.Quality)) = true;
    end
    for i=1:size(all_overl,2)
        fin_overl(i) = mean(all_overl(cov_obj(:,i),i));
    end
end
