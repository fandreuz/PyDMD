datasets = textread('../ImageSets/all.txt','%s');
datasets = cellfun(@(x) x(2:end), datasets, 'UniformOutput',false);
eval('SegTrack_config');
add_all_paths
counter = 0;
all_mq = cell(length(datasets),1);
for i=1:length(datasets)
    load(['../' datasets{i} '.mat']);
    if size(back_map,1) < size(forward_map,1)
        back_map = [back_map;zeros(size(forward_map,1) - size(back_map,1),size(back_map,2))];
    end
    mask_type = 'WithOpticalFlow';
    [directories,img_names] = parse_dataset(SVMSEGMopts.seg.imgsetpath, datasets{i});
    % From all the tracks, we only do CSI on the tracks that match ground
    % truth well, in order to save time. In principle CSI can be done on
    % any segment track.
    w = find_the_best_track(SVMSEGMopts, datasets{i},rf_obj,LinReg_obj,mask_type,back_map,20,0.9);
    % The function CSI_across_frames can accept multiple weight_grid and
    % reg_grid parameters.
    options.weight_grid = 0.3;
    options.reg_grid = 0.3;
    [mqi,best_segs] = CSI_across_frames(SVMSEGMopts, rf_obj, w, directories, img_names, mask_type, 'overlap', options);
    all_mq{i} = squeeze(max(mqi,[],3));
    % special treatment for cheetah and monkeydog, which both has objects
    % that do not go to the end of the sequence
    if strcmp(datasets{i},'cheetah')
        [forward_map, back_map, LinReg_obj, rf_obj] = track_sequence_multiple_LinReg('SegTrack_config',...
                                                   directories(1:27), img_names(1:27), 'simple', mask_type, 3000, 80, 5);
        w = find_the_best_track(SVMSEGMopts, datasets{i},rf_obj,LinReg_obj,mask_type,back_map,80,0.9);
        [mqi2,best_segs] = CSI_across_frames(SVMSEGMopts, rf_obj, w, directories(1:27), img_names(1:27), mask_type, 'overlap', options);
        all_mq{i}(2) = max(mqi2(:,:,:,2),[],3);
    end
    if strcmp(datasets{i},'monkeydog')
        [forward_map, back_map, LinReg_obj, rf_obj] = track_sequence_multiple_LinReg('SegTrack_config',...
                                                   directories(1:31), img_names(1:31), 'simple', mask_type, 3000, 80, 5);
        w = find_the_best_track(SVMSEGMopts, datasets{i},rf_obj,LinReg_obj,mask_type,back_map(:,1:30),80,0.9);
        [mqi2,best_segs] = CSI_across_frames(SVMSEGMopts, rf_obj, w, directories(1:31), img_names(1:31), mask_type, 'overlap', options);
        all_mq{i}(2) = max(mqi2(:,:,:,2),[],3);
    end    
%     for k=1:size(best_segs{1},2)
%         qtk = cell2mat(compute_segment_qualities(SVMSEGMopts.exp_dir, directories, img_names, best_segs{1}(:,k), 'trimap_15'));
%         qtk2 = cell2mat(compute_segment_qualities(SVMSEGMopts.exp_dir, directories, img_names, best_segs{1}(:,k), 'trimap_5'));
%         if size(qtk,2) > 1
%             if k==1
%                 mq_trimap15{i} = zeros(size(w,2), size(qtk,2));
%                 mq_trimap5{i} = zeros(size(w,2), size(qtk,2));
%             end
%             mq_trimap15{i}(k,:) = mean(qtk,1);
%             mq_trimap5{i}(k,:) = mean(qtk2,1);
%         else
%             mq_trimap15{i}(k) = mean(qtk,1);
%             mq_trimap5{i}(k) = mean(qtk2,1);
%         end
%     end
    % Max over tracks, mean over objects
%    mean(all_mq,3)
    counter = counter + size(mqi,4);
end
save('all_CSI.mat','all_mq');
for i=1:length(datasets)
    disp(['Dataset: ' datasets{i}]);
    for j=1:length(all_mq{i})
        disp(['Object ' int2str(j)]);
        all_mq{i}(j)
    end
end
results = cell2mat(all_mq);
disp('Mean Accuracy Per Object: ')
mean(results)
disp('Mean Accuracy Per Sequence: ')
mean_seq = compute_sequence_scores(results)