function mean_quality = segtrack_run_CSI(dataset_file_path, exp_dir, identified_tracks, lambda1, lambda2)
% segtrack_run_CSI: run CSI after segtrack_run_SPT.
% dataset_file_path and exp_dir: See README of segtrack_run_SPT.
% identified_tracks: The id of the interesting segment tracks in MySegmentTracks.
% lambda1: lambda1 in Eq. (7), controls the smoothness of the segment.
% Larger lambda1 --> more smoothness.
% lambda2: lambda2 in Eq. (7), controls the amount of optical flow
% constraints in inference. Larger lambda2 --> more optical flow
% constraints. 0 lambda2 --> no optical flow constraint.
%
%  Fuxin Li, Georgia Institute of Technology, 2013

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
   try
       load([exp_dir dataset '.mat'],'back_map','rf_obj','mask_type','lambda','LinReg_obj');
   catch E
       warning(['Error loading ' dataset '.mat. Did you run segtrack_run_SPT first?']);
       rethrow(E);
   end
   w = find_specific_tracks(SVMSEGMopts,dataset,rf_obj, LinReg_obj, mask_type, back_map, lambda, identified_tracks);
   options = [];
   if exist('lambda1','var') && ~isempty(lambda1)
    options.reg_grid = lambda1;
   end
   if exist('lambda2','var') && ~isempty(lambda2)
    options.weight_grid = lambda2;
   end
   mean_quality = CSI_across_frames(SVMSEGMopts, rf_obj, w, directories, img_names, mask_type, [], options);
end