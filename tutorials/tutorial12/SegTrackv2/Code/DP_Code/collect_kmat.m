function [K,overl] = collect_kmat(ConfigFile_or_opts, directories, img_names)
% Compute the similarity cell array, with each cell k being the similarity
% between the frame k and frame k+1. Similarity is computed with the
% chi-square kernel.
    if ~isstruct(ConfigFile_or_opts)
        eval(ConfigFile_or_opts);
    else
        SVMSEGMopts = ConfigFile_or_opts;
    end
    DefaultVal('mask_type',SVMSEGMopts.mask_type);
    masks_cur = load([SVMSEGMopts.segment_matrice_dir mask_type '/' directories{1} '/' img_names{1} '.mat']);
    masks_cur = masks_cur.masks;
    motion_feat_cur = zeros(size(masks_cur,3),2);
    [motion_feat_cur(:,1), motion_feat_cur(:,2)] = compute_centroid(masks_cur);
    K = cell(length(img_names)-1,1);
    features_cur = get_features(SVMSEGMopts.tracking_features, SVMSEGMopts.measurement_dir, ...
    SVMSEGMopts.scaling_types, mask_type, directories{1}, img_names{1});
    for i=1:length(img_names)-1
        if (mod(i,10)==0)
            i
        end
        features_next = get_features(SVMSEGMopts.tracking_features, SVMSEGMopts.measurement_dir, ...
            SVMSEGMopts.scaling_types, mask_type, directories{i+1}, img_names{i+1});
        masks_next = load([SVMSEGMopts.segment_matrice_dir mask_type '/' directories{i} '/' img_names{i+1} '.mat']);
        masks_next = masks_next.masks;        
        % Compute the chi square kernel of features_cur and features_next
        K{i} = chi_square_kernel(features_cur{1}', features_next{1}', 1.5) + chi_square_kernel(features_cur{2}', features_next{2}', 1.5);
        motion_feat_next = zeros(size(masks_next,3),2);
        [motion_feat_next(:,1), motion_feat_next(:,2)] = compute_centroid(masks_next);        
        overlay_mat = false(size(masks_next,3),size(masks_cur,3));
        for j=1:size(masks_cur,3)
            % Find the segments that stay within motion constraints
            overlay_mat(:,j) = sum(abs(bsxfun(@minus, motion_feat_cur(j,:), motion_feat_next))>0.15,2) == 0;
        end
        % Only connect the ones that are possible, measured by centroid
        % displacement
        K{i} = K{i} .* overlay_mat';
        K{i} = K{i} / 2;
        overl{i} = fast_segm_overlap_mex(masks_cur);
        features_cur = features_next;
        masks_cur = masks_next;
        motion_feat_cur = motion_feat_next;
    end
    overl{length(img_names)} = fast_segm_overlap_mex(masks_cur);
end