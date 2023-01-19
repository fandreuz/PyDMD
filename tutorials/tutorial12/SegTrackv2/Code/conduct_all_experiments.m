%  Fuxin Li, Georgia Institute of Technology, 2013
% Read in the dataset file in the SegTrack format
datasets = textread('./ImageSets/all.txt','%s');
datasets = cellfun(@(x) x(2:end), datasets, 'UniformOutput',false);
% Evaluate the ConfigFile to generate SVMSEGMopts
eval('SegTrack_config');
lambda = 100;
% Add the necessary paths
add_all_paths
for i=1:length(datasets)
    % The script would not overwrite result files, if they are already
    % there. Change this if needed
    if exist(['./' datasets{i} '.mat'],'file')
        continue;
    end
    mask_type = 'WithOpticalFlow';
    % Run CPMC first, this script will do the flow/boundary/CPMC
    % computation
    run_cpmc_dataset(SVMSEGMopts.exp_dir, SVMSEGMopts.seg.imgsetpath, datasets{i}, mask_type);
    % Parse the dataset to generate a list of sub-directories and image
    % names
    [directories,img_names] = parse_dataset(SVMSEGMopts.seg.imgsetpath, datasets{i});
    % Compute the overlap score between the segments and the ground truth,
    % skip this line if there's no ground truth
    SegTrack_segment_quality_all(SVMSEGMopts, datasets{i}, [], 'WithOpticalFlow');
%    rank_and_resample(SVMSEGMopts,directories, img_names, 'trainval_trees', 'WithOpticalFlow_400', 400);
    
    
%   img_names = img_names(1:4);
%   directories = directories(1:4);
    % Compute the color SIFT features
    compute_color_sift_feature(SVMSEGMopts, directories, img_names, mask_type);
    
%    SegTrack_segment_quality_all(SVMSEGMopts, datasets{i}, [], mask_type);
    % Main function, multi-regression tracking use the computed features and segments
    % Outputs forward_map: the map from time t to time t+1
    % back_map: the map from t+1 to t
    %
    % For example, forward_map(3,t) = 50 means segment #3 at time t is mapped
    % to segment #50 at time t+1
    % back_map(50,t) = 3 means segment #50 at time t+1 is mapped to segment
    % #3 at time t
    % for some segment k that is not tracked back_map(k,t) is 0
    %
    % LinReg_obj: the regression object storing the covariance and
    % input-output correlation matrices
    % rf_obj: the random Fourier feature object
        [forward_map, back_map, LinReg_obj, rf_obj] = track_sequence_multiple_LinReg('SegTrack_config',...
                                                   directories, img_names, 'simple', mask_type, 3000, lambda, 5);
    % Save the result file, these are all we need
    save(['./' datasets{i} '.mat'], 'forward_map','back_map','LinReg_obj','rf_obj','mask_type','lambda');
    % Plot tracks using the backward mapping
%     qual_track = plot_tracks_from_back_map(SVMSEGMopts.exp_dir, directories, img_names, mask_type,...
%                                            back_map, 'overlap', []);
%     % CSI is not called in this script, see find_CSI_parms.m for a wrapper
%     % on that part
%     if iscell(qual_track)
%         disp(['Dataset: ' datasets{i}]);
%         for j=1:length(qual_track)
%             disp(['Object ' int2str(j)]);
%             max(mean(qual_track{j},2))
%         end
%     else
%         disp(['Dataset: ' datasets{i}]);
%         max(mean(qual_track,2))
%     end
end
print_all_results
