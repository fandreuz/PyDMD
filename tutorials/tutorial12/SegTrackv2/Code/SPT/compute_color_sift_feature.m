function compute_color_sift_feature(ConfigFile_or_opts, directories, img_names, mask_type, overwrite)
    if ~isstruct(ConfigFile_or_opts)
        eval(ConfigFile_or_opts);
    else
        SVMSEGMopts = ConfigFile_or_opts;
    end
    DefaultVal({'mask_type','overwrite'},{SVMSEGMopts.mask_type,false});
    for i=1:length(img_names)
        SvmSegm_extract_all_class_features_nofiles(SVMSEGMopts.exp_dir, [directories{i} '/' img_names{i}], mask_type, {'bow_dense_color_sift_3_scales_figure_300','bow_dense_color_sift_3_scales_ground_300'},true,overwrite);
    end
end