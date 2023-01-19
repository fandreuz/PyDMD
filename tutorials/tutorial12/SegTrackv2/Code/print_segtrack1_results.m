datasets = {'girl','birdfall','parachute','cheetah','monkeydog','penguin'};
eval('SegTrack_config');
add_all_paths;
mask_type = 'WithOpticalFlow';
clear all_overl
for i=1:length(datasets)
    SVMSEGMopts.segment_quality_type = 'pixel_difference';
%    SegTrack_segment_quality_all(SVMSEGMopts, datasets{i}, [], 'WithOpticalFlow',[],true);
    all_overl{i} = SegTrack_find_best_qual(SVMSEGMopts,  datasets{i},'pixel_difference', mask_type, true);
    all_overl{i} = all_overl{i}(1);
    load(['../' datasets{i} '.mat']);
    [directories, img_names] = parse_dataset(SVMSEGMopts.seg.imgsetpath, datasets{i});
    if size(back_map,1) < size(forward_map,1)
        back_map = [back_map;zeros(size(forward_map,1) - size(back_map,1),size(back_map,2))];
    end    
    SVMSEGMopts.segment_quality_type = 'overlap';
    w = find_the_best_track(SVMSEGMopts, datasets{i},rf_obj,LinReg_obj,mask_type,back_map,80,0.9);
    SVMSEGMopts.segment_quality_type = 'pixel_difference';
    [qual_track,~,num_tracks(i), num_segs(i)] = plot_tracks_from_back_map(SVMSEGMopts.exp_dir, directories, img_names, mask_type,...
                                           back_map, 'pixel_difference', [],true);
    [mqi_CSI{i},best_segs] = CSI_across_frames(SVMSEGMopts, rf_obj, w, directories, img_names, mask_type, 'pixel_difference');
%     trimap_15 = plot_tracks_from_back_map(SVMSEGMopts.exp_dir, directories, img_names, mask_type,...
%                                            back_map, 'trimap_15', [],true);
%     trimap_5 = plot_tracks_from_back_map(SVMSEGMopts.exp_dir, directories, img_names, mask_type,...
%                                            back_map, 'trimap_5', [],true);                                       
    if iscell(qual_track)
        disp(['Dataset: ' datasets{i}]);
        j = 1;
        min(mean(qual_track{j},2))
    else
        disp(['Dataset: ' datasets{i}]);
        min(mean(qual_track,2))
    end
%     if iscell(qual_track)
%         for j=1:length(qual_track)
%             disp(['Trimap-15: Object ' int2str(j)]);
%             min(mean(trimap_15{j},2))
%             disp(['Trimap-5: Object ' int2str(j)]);
%             min(mean(trimap_5{j},2))
%         end
%     else
%         disp('Trimap-15:');
%         min(mean(trimap_15,2))
%         disp('Trimap-5:');
%         min(mean(trimap_5,2))
%     end
end
disp('Average number of tracks: ');
mean(num_tracks)
disp('Average number of segments: ');
mean(num_segs)

all_overl = cell2mat(all_overl);
save('SegTrack1.mat','all_overl','qual_track','mqi_CSI','num_tracks','num_segs');