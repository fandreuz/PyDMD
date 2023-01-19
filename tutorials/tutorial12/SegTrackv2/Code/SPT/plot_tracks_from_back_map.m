function [qual_track,track_scores, num_tracks, num_segs] = plot_tracks_from_back_map(exp_dir, directories, img_names, mask_type, back_map, quality_type, back_score, no_plot, merge_plots)
    DefaultVal({'no_plot','merge_plots'},{false,[]});
    back_tracker = find(back_map(:,end)~=0);
    track_scores = zeros(length(back_tracker),1);
    qual_track = zeros(length(back_tracker), length(img_names));
    back_map = [zeros(1,length(img_names)-1);back_map];
    avg_segs = zeros(length(img_names),1);
    if ~no_plot
        for i=1:length(img_names)
            try
                rmdir([exp_dir 'MySegmentTracks/' mask_type '/' directories{i} '/'],'s');
            end
        end
    end
    for i=length(img_names):-1:1
        img_ext = get_img_extension([exp_dir 'JPEGImages/' directories{i} '/' img_names{i}]);
        I = imread([exp_dir 'JPEGImages/' directories{i} '/' img_names{i} img_ext]);
        var = load([exp_dir 'MySegmentsMat/' mask_type '/' directories{i} '/' img_names{i} '.mat']);
        masks = var.masks;
        if nargout > 1 && exist('back_score','var') && ~isempty(back_score) && i > 1
            track_scores = track_scores + back_score(back_tracker,i-1);
        end
        if ~no_plot
            if ~isempty(merge_plots)
                cmap = SvmSegm_labelcolormap(255);
                merged_img = I;
            end
            for j=1:length(back_tracker)
                track_dir = [exp_dir 'MySegmentTracks/' mask_type '/' directories{i} '/' int2str(j) '/'];
                if ~exist(track_dir,'dir')
                    mkdir(track_dir);
                end
                if back_tracker(j) == 0
                    continue;
                end
                if isempty(merge_plots)
                    merged_img = merge_img(masks(:,:,back_tracker(j)), I);
                    cmap = [0 0 0;1 1 1];
                    imwrite(masks(:,:,back_tracker(j)), cmap, [track_dir num2str(i) '.png']);
                    imwrite(merged_img, [track_dir num2str(i) '.jpg']);
                else
                    if ismember(j, merge_plots)
                        merged_img = merge_img(masks(:,:,back_tracker(j)), merged_img, cmap(j+1,:),0.6);
                    end
                end
            end
            if ~isempty(merge_plots)
                track_dir = [exp_dir 'MySegmentTracks/' mask_type '/' directories{i} '/Merged/'];
                if ~exist(track_dir,'dir')
                    mkdir(track_dir);
                end
                imwrite(merged_img, [track_dir num2str(i) '.jpg']);
            end
        end
        if nargout > 0 && exist('quality_type','var') && ~isempty(quality_type)
            still_tracking = back_tracker ~= 0;
            try
            qual_track_this = compute_segment_qualities(exp_dir, directories(i), img_names(i), {masks(:,:,back_tracker(still_tracking))},quality_type);
            avg_segs(i) = size(masks,3);
            if length(qual_track_this) > 1
                if ~iscell(qual_track)
                    qual_track = {qual_track};
                end
                for j=1:length(qual_track_this)
                    qual_track{j}(still_tracking,i) = qual_track_this{j};
                end
            else
                qual_track(still_tracking,i) = qual_track_this{1};
            end
            catch
                qual_track = [];
            end
        end
        if i>1
            back_tracker = back_map(back_tracker+1,i-1);
        end
    end
    if nargout > 1
        num_tracks = length(back_tracker);
        num_segs = mean(avg_segs);
    end
end
