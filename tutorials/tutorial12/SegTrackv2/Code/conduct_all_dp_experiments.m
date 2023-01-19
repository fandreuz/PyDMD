% Run all the experiments using the Dynamic Programming tracking algorithm
% in Hamed Pirsiavash, Deva Ramanan, Charless Fowlkes, "Globally-Optimal 
% Greedy Algorithms for Tracking a Variable Number of Objects," Computer 
% vision and Pattern Recognition CVPR 2011
datasets = textread('../ImageSets/all.txt','%s');
datasets = cellfun(@(x) x(2:end), datasets, 'UniformOutput',false);
eval('SegTrack_config');
add_all_paths
counter = 1;
for i=1:length(datasets)
    % Don't overwrite
    [directories,img_names] = parse_dataset(SVMSEGMopts.seg.imgsetpath, datasets{i});
    mask_type = 'WithOpticalFlow';
    if ~exist([datasets{i} '_dp.mat'],'file')
    % Get rid for now to avoid verbose outputs on screen
%    SegTrack_segment_quality_all(SVMSEGMopts, datasets{i}, [], 'WithOpticalFlow');
%    rank_and_resample(SVMSEGMopts,directories, img_names, 'trainval_trees', 'WithOpticalFlow_400', 400);
%    compute_color_sift_feature(SVMSEGMopts, directories, img_names, mask_type);
    [K,overl_all] = collect_kmat(SVMSEGMopts,directories,img_names);
    inds_all = mytracking_dp(K,overl_all);
    save([datasets{i} '_dp.mat'], 'inds_all');
    else
        load([datasets{i} '_dp.mat']);
    end
    qual_track = get_inds_quality(SVMSEGMopts.exp_dir, directories, img_names, 'WithOpticalFlow', inds_all,'overlap');
%    qual_track = get_inds_quality(SVMSEGMopts.exp_dir, directories, img_names, 'WithOpticalFlow', inds_all,'band_overlap');
    if iscell(qual_track)
        disp(['Dataset: ' datasets{i}]);
        for j=1:length(qual_track)
            disp(['Object ' int2str(j)]);
            err(counter) = max(mean(qual_track{j},2))
            counter = counter + 1;
        end
    else
        disp(['Dataset: ' datasets{i}]);
        err(counter) = max(mean(qual_track,2))
        counter = counter + 1;
    end
    num_tracks(i) = size(inds_all,2);
end
