clear
datasets = textread('./ImageSets/all.txt','%s');
datasets = cellfun(@(x) x(2:end), datasets, 'UniformOutput',false);
eval('SegTrack_config');
add_all_paths;
mask_type = 'WithOpticalFlow';
clear all_overl
counter = 1;
% All SegTrack v2 objects
results = zeros(24,1);
for i=1:length(datasets)
    all_overl{i} = SegTrack_find_best_qual(SVMSEGMopts,  datasets{i},'overlap', mask_type);
    load(['./' datasets{i} '.mat'],'back_map','forward_map');
    [directories, img_names] = parse_dataset(SVMSEGMopts.seg.imgsetpath, datasets{i});
    if size(back_map,1) < size(forward_map,1)
        back_map = [back_map;zeros(size(forward_map,1) - size(back_map,1),size(back_map,2))];
    end    
    [qual_track,~,num_tracks(i), num_segs(i)] = plot_tracks_from_back_map(SVMSEGMopts.exp_dir, directories, img_names, mask_type,...
                                           back_map, 'overlap', [],true);
%     trimap_15 = plot_tracks_from_back_map(SVMSEGMopts.exp_dir, directories, img_names, mask_type,...
%                                            back_map, 'trimap_15', [],true);
%     trimap_5 = plot_tracks_from_back_map(SVMSEGMopts.exp_dir, directories, img_names, mask_type,...
%                                            back_map, 'trimap_5', [],true);            
%  Special treatment for Cheetah-Cheetah and Monkeydog-Dog, who left the
%  scene early.
    if strcmp(datasets{i},'cheetah')
        qual_track2 = plot_tracks_from_back_map(SVMSEGMopts.exp_dir, directories(1:27), img_names(1:27), mask_type,...
                                           back_map(:,1:26), 'overlap', [],true);
        qual_track{2} = qual_track2{2};
    end
    if strcmp(datasets{i},'monkeydog')
        qual_track2 = plot_tracks_from_back_map(SVMSEGMopts.exp_dir, directories(1:31), img_names(1:31), mask_type,...
                                           back_map(:,1:30), 'overlap', [],true);
        qual_track{2} = qual_track2{2};
    end
    if iscell(qual_track)
        disp(['Dataset: ' datasets{i}]);
        for j=1:length(qual_track)
            disp(['Object ' int2str(j)]);
            results(counter) = max(mean(qual_track{j},2));
            results(counter)
            counter = counter + 1;
        end
    else
        disp(['Dataset: ' datasets{i}]);
        results(counter) = max(mean(qual_track,2));
        results(counter)
        counter = counter + 1;
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
disp('Mean Accuracy Per Object: ')
mean(results)
disp('Mean Accuracy Per Sequence: ')
mean_seq = compute_sequence_scores(results)
all_overl = cell2mat(all_overl);
disp('CPMC Best: ');
all_overl
