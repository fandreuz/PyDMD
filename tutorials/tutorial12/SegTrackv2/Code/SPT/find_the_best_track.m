function w = find_the_best_track(ConfigFile_or_opts, dataset, rf_obj, LinReg_obj, mask_type, back_map, lambda, cut_off)
% Locate the best tracks within all the LinReg_objs and output the weights
% of those
    if ~isstruct(ConfigFile_or_opts)
        eval(ConfigFile_or_opts);
    else
        SVMSEGMopts = ConfigFile_or_opts;
    end
    [directories, img_names] = parse_dataset(SVMSEGMopts.seg.imgsetpath, dataset);
    if length(directories) > size(back_map,2) +1
        directories = directories(1:size(back_map,2)+1);
        img_names = img_names(1:size(back_map,2)+1);
    end
    DefaultVal({'mask_type','lambda','cut_off'}, {SVMSEGMopts.mask_type, 10000, 0.7});
    qual_track = plot_tracks_from_back_map(SVMSEGMopts.exp_dir, directories, img_names, mask_type, back_map, 'overlap', [], true);
    to_keep = false(size(qual_track,1),1);
    w = [];
    if iscell(qual_track)
        for i=1:length(qual_track)
            mq = mean(qual_track{i},2);
            max_mqi = max(mq);
            to_keep = to_keep | mq > max_mqi * cut_off;
        end
    else
        mq = mean(qual_track,2);
        max_mqi = max(mq);
        to_keep = to_keep | mq > max_mqi * cut_off;
    end
    % The sequence from back_map is sorted by this
    back_tracker = find(back_map(:,end)~=0);
    to_locate = back_tracker(to_keep);    
% Now test the LinReg_objs on the final frame to determine which one correspond to which
    features_fin = get_features(SVMSEGMopts.tracking_features, SVMSEGMopts.measurement_dir, ...
     SVMSEGMopts.scaling_types, mask_type, directories{end}, img_names{end});
    XX1 = rf_featurize(rf_obj, features_fin{1}');
    XX2 = rf_featurize(rf_obj, features_fin{2}');
    allX_test = [XX1 XX2];
    for i=1:length(LinReg_obj)
        if LinReg_obj{i}.no_target()
            continue;
        end
        this_w = LinReg_obj{i}.Regress(lambda);
    %     pred{i} = zeros(size(allX_test,1),size(w{i},2));
    %     for j=1:size(w{i},2)
    %         pred{i}(overlay_mat(:,j),j) = w{i}(1,j) + allX_test(overlay_mat(:,j),:) * w{i}(2:end,j);
    %     end
        pred = bsxfun(@plus, this_w(1,:), allX_test * this_w(2:end,:));
        [~,b] = max(pred,[],1);
        [~,ia] = intersect(b,to_locate);
        w = [w this_w(:,ia)];
    end
end