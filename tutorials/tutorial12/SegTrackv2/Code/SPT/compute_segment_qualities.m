function quals = compute_segment_qualities(exp_dir, directories, img_names, masks, quality_type)
% Compute segment qualities of the segments in masks by matching it with
% the ground truth specified in [exp_dir 'GroundTruth/']
% quality_type can be: 
%                     pixel_difference, compute the number of different
%                                       pixels between the masks and GT
%                     overlap,          compute the intersection-over-union 
%                                       overlap score
%                     trimap_k,         compute the TriMap measure on a
%                                       band of k pixels
%                     band_overlap_k,   compute the overlap measure but
%                                       only on a band of k pixels
%                                       surrounding the ground truth object
%                     intersect_gt,     compute the percentage of GT pixels
%                                       that are within the segment
%                     centroid_displacement, compute the l2 distance
%                                       between the centroid of GT 
    quals = cell(length(img_names),1);
    for i=1:length(directories)
        gt_base_dir = [exp_dir 'GroundTruth/' directories{i} '/'];
    try
        img_ext = get_img_extension([gt_base_dir img_names{i}]);
        GT = imread([gt_base_dir img_names{i} img_ext]);
        GT = GT(:,:,1);
        if strcmp(quality_type,'pixel_difference')
            quals{i} = compute_overlaps(masks{i}, GT > 128, [],quality_type);    
        elseif strncmp(quality_type, 'trimap_',7)
            trimap_num = sscanf(quality_type, 'trimap_%d');
            if isempty(trimap_num) || trimap_num == 0
                error('When using trimap, must specify a number! For example trimap_7, trimap_15, trimap_20, etc.');
            end
            quals{i} = 1 - compute_overlaps(masks{i}, GT > 128, imdilate(bwperim(GT > 128), ones(trimap_num,trimap_num)), 'pixel_percentage');
        elseif strncmp(quality_type, 'band_overlap_',13)
            band_num = sscanf(quality_type, 'band_overlap_%d');
            if isempty(band_num) || band_num == 0
                error('When using band_overlap, must specify a number! For example band_overlap_7, band_overlap_15, band_overlap_20, etc.');
            end
            band_masks = zeros(size(masks{i}));
            for j=1:size(masks{i},3)
                band_masks(:,:,j) = imdilate(bwperim(masks{i}(:,:,j)), ones(band_num,band_num));
            end
            quals{i} = compute_overlaps(band_masks, imdilate(bwperim(GT > 128), ones(band_num,band_num)),[], 'overlap');
        else
            quals{i} = compute_overlaps(masks{i}, GT > 224, GT > 224 | GT < 32,quality_type);
        end
    catch
        for j=1:254
            try
                img_ext = get_img_extension([gt_base_dir int2str(j) '/' img_names{i}]);
                GT = imread([gt_base_dir int2str(j) '/' img_names{i} img_ext]);
                GT = GT(:,:,1);
        if strcmp(quality_type,'pixel_difference')
            quals{i,j} = compute_overlaps(masks{i}, GT > 128, [],quality_type);    
        elseif strncmp(quality_type, 'trimap_',7)
            trimap_num = sscanf(quality_type, 'trimap_%d');
            if isempty(trimap_num) || trimap_num == 0
                error('When using trimap, must specify a number! For example trimap_7, trimap_15, trimap_20, etc.');
            end
            quals{i,j} = 1 - compute_overlaps(masks{i}, GT > 128, imdilate(bwperim(GT > 128), ones(trimap_num,trimap_num)), 'pixel_percentage');            
        elseif strncmp(quality_type, 'band_overlap_',13)
            band_num = sscanf(quality_type, 'band_overlap_%d');
            if isempty(band_num) || band_num == 0
                error('When using band_overlap, must specify a number! For example band_overlap_7, band_overlap_15, band_overlap_20, etc.');
            end
            band_masks = zeros(size(masks{i}));
            for k=1:size(masks{i},3)
                band_masks(:,:,k) = imdilate(bwperim(masks{i}(:,:,k)), ones(band_num,band_num));
            end
            quals{i,j} = compute_overlaps(band_masks, imdilate(bwperim(GT > 128), ones(band_num,band_num)), [], 'overlap');
        else
            quals{i,j} = compute_overlaps(masks{i}, GT > 224, GT > 224 | GT < 32,quality_type);
        end
            catch
                break;
            end
        end
    end
    end
end