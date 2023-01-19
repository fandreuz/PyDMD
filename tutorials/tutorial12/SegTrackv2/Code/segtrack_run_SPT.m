function mean_quality = segtrack_run_SPT(dataset_file_path, exp_dir, mask_type, tracking_only, Napp, lambda, restart_frames, force_recompute)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Segtrack_package_version: This performs the least-squares based Segment
% Pool Tracking algorithm in the ICCV 2013 paper. It does
% not include the inference part (CSI inference) in Section 4. The CSI
% is called separately in CSI_across_frames or find_CSI_parms.
%
% Parameters:
%   dataset_file_path: The path to a text file supplying the names and order of the images in the sequence
%   The text file should have the following format:
%   First line: folder_name/
%   From second line on: each line should contain the name of a frame, sorted in ascending time order. e.g.
%   frame_001
%   frame_002
%   frame_003
%   etc.
%   See examples in the ../ImageSets/ folder.
%
%   exp_dir: The starting data directory. The algorithm will generate data
%            items under this directory.
% Optional parameters:
%   mask_type: The name of the segmenter (whatever string you prefer,
%   default WithOpticalFlow)
%  tracking_only: optional, if true, only run the tracking part.
%   Napp: Random Fourier dimensions (default 3000)
%   lambda: Regularization for ridge regression (default 80)
%   restart_frames: Start segment tracks from the first X frames (default
%   5)
%   force_recompute: By default, segtrack_run_all will compute the optical
%   flow/boundary/segment/features only once and load them from files
%   afterwards. This option will force segtrack_run_all to recompute all of
%   them.
%  Output:
%           mean_quality: the average overlap of the segment track if
%           GroundTruth files are present.
%
%  Fuxin Li, Georgia Institute of Technology, 2013
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Add / to exp_dir.
if exp_dir(end) ~= '/' && exp_dir(end) ~= '\'
    exp_dir = [exp_dir '/'];
end
[directories,img_names] = parse_dataset(dataset_file_path);
dataset_pos = find(dataset_file_path == '/' | dataset_file_path =='\',1, 'last');
dataset = dataset_file_path(dataset_pos+1:end);
% Remove .txt suffix, if exists
if strcmp(dataset(end-3:end),'.txt')
    dataset = dataset(1:end-4);
end
SVMSEGMopts = Generate_configs(exp_dir);
%SegTrack_segment_quality_all(SVMSEGMopts, datasets{i}, [], 'WithOpticalFlow');
%    rank_and_resample(SVMSEGMopts,directories, img_names, 'trainval_trees', 'WithOpticalFlow_400', 400);
    DefaultVal({'mask_type','Napp','lambda','restart_frames','tracking_only','force_recompute'}, {'WithOpticalFlow',3000,100,5,false,false});
    if force_recompute
        run_cpmc_dataset(exp_dir, dataset_file_path, [], mask_type,true);
        compute_color_sift_feature(SVMSEGMopts, directories, img_names, mask_type,true);
    else
        if ~tracking_only
            run_cpmc_dataset(exp_dir, dataset_file_path, [], mask_type);
            compute_color_sift_feature(SVMSEGMopts, directories, img_names, mask_type);
        end
    end
    [forward_map, back_map, LinReg_obj, rf_obj,w] = track_sequence_multiple_LinReg(SVMSEGMopts,...
                                                   directories, img_names, 'simple', mask_type,Napp, lambda, restart_frames);
    if size(back_map,1) < size(forward_map,1)
        back_map = [back_map;zeros(size(forward_map,1) - size(back_map,1),size(back_map,2))];
    end                                                   
    save([exp_dir dataset '.mat'], 'forward_map','back_map','LinReg_obj','rf_obj','mask_type','lambda','w');
    mean_quality = plot_tracks_from_back_map(exp_dir, directories, img_names, mask_type,...
                                           back_map, 'overlap', []);
end