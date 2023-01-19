function [forward_map, back_map, LinReg_obj, rf_obj, w] = track_sequence_multiple_LinReg(ConfigFile_or_opts, directories, img_names, inference_type, mask_type, Napp, reg_param, restart_frames)
% Minimal call is: track_sequence_multiple_LinReg(ConfigFile_or_opts, directories, img_names)
% Parameters:
% Output:
%          forward_map: Each row correspond to a segment id, each column correspond to a frame. For row i and column j, forward_map stores the matched segment in frame j+1 for the i-th segment in frame j.
%             back_map: Similar to forward_map, but backwards. For row i and column j, back_map stores the matched segment in frame j for the i-th segment in frame j+1.
%                       back_map is used in the function plot_tracks_from_back_map which plots the segment tracks in the MySegmentTracks folder.
%           LinReg_obj: The stored H and C matrices for all the segment tracks. This can be used if one wants to continue tracking, or re-do the regression with a different regularization.
%               rf_obj: The random feature object specifying the feature embedding. This is needed when generating new testing samples (which need to be embedded as well).
%                    w: The regression weight vectors.
%  Fuxin Li, Georgia Institute of Technology, 2013
    if ~isstruct(ConfigFile_or_opts)
        eval(ConfigFile_or_opts);
    else
        SVMSEGMopts = ConfigFile_or_opts;
    end
    DefaultVal({'inference_type','mask_type','Napp','reg_param','restart_frames'},{'simple',SVMSEGMopts.mask_type,3000,80,2});

    options.method = 'direct';
    options.Nperdim = 5;
    options.single = true;
    features_cur = get_features(SVMSEGMopts.tracking_features, SVMSEGMopts.measurement_dir, ...
        SVMSEGMopts.scaling_types, mask_type, directories{1}, img_names{1});
    options.params = decide_parameter_list(cell2mat(features_cur),20);
    rf_obj = InitExplicitKernel('exp_chi2',1.5,300,Napp,options);
    masks_cur = load([SVMSEGMopts.segment_matrice_dir mask_type '/' directories{1} '/' img_names{1} '.mat']);
    masks_cur = masks_cur.masks;
    new_trackid = cell(1,1);
    new_trackid{1} = 1:size(masks_cur,3);
    LinReg_obj = [];
    motion_feat_cur = zeros(size(masks_cur,3),2);
    [motion_feat_cur(:,1), motion_feat_cur(:,2)] = compute_centroid(masks_cur);
    old_unique_segs = 1:size(masks_cur,3);
    weights = ones(size(masks_cur,3),1);
    for i=1:length(img_names)-1
        i
        new_trackid
        if directories{i} ~= directories{i+1}
            disp('Currently does not support simultaneous tracking multiple sequences yet! Will add later.');
        end

        masks_next = load([SVMSEGMopts.segment_matrice_dir mask_type '/' directories{i} '/' img_names{i+1} '.mat']);
        masks_next = masks_next.masks;
        motion_feat_next = zeros(size(masks_next,3),2);
        [motion_feat_next(:,1), motion_feat_next(:,2)] = compute_centroid(masks_next);

        overlay_mat = false(size(masks_next,3),length(old_unique_segs));
        for j=1:length(old_unique_segs)
            % Find the segments that stay within motion constraints
            overlay_mat(:,j) = sum(abs(bsxfun(@minus, motion_feat_cur(old_unique_segs(j),:), motion_feat_next))>0.15,2) == 0;
        end
        % Overl_mat will be the output

        t = tic();
        overl_mat = fast_segm_overlap_mex(masks_cur(1:2:end,1:2:end,:), masks_cur(1:2:end,1:2:end,old_unique_segs));
        disp('Overlap time: ');
        toc(t);
        %        end
        % Load input features
        features_next = get_features(SVMSEGMopts.tracking_features, SVMSEGMopts.measurement_dir, ...
            SVMSEGMopts.scaling_types, mask_type, directories{i+1}, img_names{i+1});
        t = tic();
        [pred, w, LinReg_obj] = do_traintest_1frame(features_cur, features_next, overl_mat, reg_param, rf_obj, LinReg_obj, new_trackid,weights);
        disp('Train_test time: ');
        toc(t);

        ranges = [0 cumsum(cellfun(@length,new_trackid))];
        mat_trackid = cell2mat(new_trackid);
        if strcmp(inference_type, 'simple')
            a = cell(1,length(LinReg_obj));
            b = cell(1,length(LinReg_obj));
            weights = zeros(size(masks_next,3),length(LinReg_obj));
            for j=1:length(LinReg_obj)
                if ~isempty(pred{j})
                    [a{j},b{j}] = max(pred{j} .* overlay_mat(:,new_trackid{j}));
                    weights(:,j) = max(pred{j} .* overlay_mat(:,new_trackid{j}),[],2);
                else
                    a{j} = single([]);
                end
            end
            % Everything together
            a = cell2mat(a);
            forward_assignment = cell2mat(b);
            [c,d] = sort(a,'descend');
            % Cut ones with really low scores
            cut_plc = find(c>=0.1,1,'last');
            [unique_segs,m] = unique(forward_assignment(d(1:cut_plc)),'first');
            backward_assignment = d(m);
        elseif strcmp(inference_type,'hungarian')
            % Still use overlay
            weights = zeros(size(masks_next,3),length(LinReg_obj));
            for j=1:length(pred)
                if ~isempty(new_trackid{j})
                    pred{j} = pred{j} .* overlay_mat(:,new_trackid{j});
                    weights(:,j) = max(pred{j} .* overlay_mat(:,new_trackid{j}),[],2);
                else
                    pred{j} = single([]);
                end
            end
            pred_all = 1 -cell2mat(pred');
            % Just increase the ones that we don't want to match
            pred_all(pred_all >= 0.8) = inf;
            pred_all(pred_all < 0) = 0;
            [forward_assignment, cost] = assignmentoptimal(double(pred_all)');
            % backward_assignment can be done by this because Hungarian
            % matching will automatically be unique, other than 0 (no
            % match)
            [unique_segs,backward_assignment] = unique(forward_assignment);
            unique_segs = unique_segs';
            if unique_segs(1) == 0
                unique_segs = unique_segs(2:end);
                backward_assignment = backward_assignment(2:end);
            end
            inds = sub2ind(size(pred_all), unique_segs, mat_trackid(backward_assignment));
%            scores = 1 - pred_all(inds);
        end
        forward_map(old_unique_segs(mat_trackid),i) = forward_assignment;
        back_map(unique_segs,i) = old_unique_segs(mat_trackid(backward_assignment));
%        back_scores(unique_segs,i) = scores;
        for j=1:length(LinReg_obj)
            dm_inrange = backward_assignment > ranges(j) & backward_assignment <= ranges(j+1);
            new_trackid{j} = unique_segs(dm_inrange);
            tracks_tokeep = backward_assignment - ranges(j);
            tracks_tokeep = tracks_tokeep(dm_inrange);
            LinReg_obj{j}.prune_targets(tracks_tokeep);
        end
        % Do the "leftover routine" only on the first half of the sequence
        if i < restart_frames
            % Find the "leftovers"
            leftovers = setdiff(1:size(masks_next,3),unique_segs);
            if ~isempty(leftovers)
                % Then start a new LinReg_obj on those leftover segments
                LinReg_obj{j+1} = [];
                new_trackid{j+1} = leftovers;
            end
            features_cur = features_next;
            masks_cur = masks_next;
            motion_feat_cur = motion_feat_next;
            old_unique_segs = 1:size(masks_next,3);
            weights = [weights ones(size(masks_next,3), 1)];
        else
            % Otherwise trim the tracks
%            features_cur = cellfun(@(x) x(:,unique_segs), features_next,'UniformOutput',false);
%            masks_cur = masks_next(:,:,unique_segs);
            masks_cur = masks_next;
            features_cur = features_next;
            for j=1:length(LinReg_obj)
                [~,new_trackid{j}] = find(bsxfun(@eq, new_trackid{j}',unique_segs));
                new_trackid{j} = new_trackid{j}';
            end
%            motion_feat_cur = motion_feat_next(unique_segs,:);
            motion_feat_cur = motion_feat_next;
            old_unique_segs = unique_segs;
        end
    end
end
