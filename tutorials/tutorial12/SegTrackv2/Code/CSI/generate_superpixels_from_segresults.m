function [sup_size, sup_assignment, scores, bag_ids, Pixels] = generate_superpixels_from_segresults(masks, scores, bag_ids, score_cutoff, size_cutoff, visualize)
% For one image, generate superpixels from segmentation results.
% Output the results by returning argument, no savefiles.
    if ~exist('size_cutoff','var')
        size_cutoff = 50;
    end
    if ~exist('visualize','var')
        visualize = false;
    end
    if ~exist('bag_ids','var') || isempty(bag_ids)
        bag_ids = (1:size(scores,1))';
    end
    max_scores = max(scores,[],2);
    if visualize
        cmap = SvmSegm_labelcolormap(255);
    end
        to_keep = max_scores > score_cutoff;
        masks = masks(:,:,to_keep);
        bag_ids = bag_ids(to_keep);
        scores = scores(to_keep,:);
        sz = size(masks);
        if length(sz)==2
            sz(3) = 1;
        end
        [masks2, idx2] = assign_superpix(reshape(masks,sz(1)*sz(2),sz(3))');
        masks2 = masks2';
%        [masks2,~,idx2] = unique(reshape(masks,sz(1)*sz(2),sz(3)), 'rows');
        % No segments
        if isempty(masks2)
            Pixels = [];
            sup_assignment = [];
            sup_size = [];
            return;
        end
        idx3 = reshape(idx2, sz(1), sz(2));
        [~, ~, edgelet_sp] = generate_sp_neighborhood(idx3);
        D = sparse(double(edgelet_sp(:,1)), double(edgelet_sp(:,2)), ones(size(edgelet_sp,1),1), size(masks2,1), size(masks2,1));
        D = D + D';
        hs = histc(idx2,1:size(masks2,1));
        [num_pixels, sort_idx] = sort(hs,'descend');
        D = D(sort_idx, sort_idx);
        [small_assign,simple_ones] = assign_small_patches(masks2(sort_idx,:), num_pixels >= size_cutoff, D);
        sup_assignment = masks2(sort_idx(simple_ones),:);
        sup_size = num_pixels(simple_ones);
%        sort_idx = sort_idx(1:select);
        Pixels = zeros(size(masks,1),size(masks,2));
        % re-sort it to make the index nicer
        for j=1:max(small_assign)
            Pixels(ismember(idx2,sort_idx(small_assign==j))) = j;
        end
        if visualize
            imshow(Pixels,cmap);
        end
end

function [small_assign,simple_ones] = assign_small_patches(masks, to_keep, D)
    small_assign = zeros(length(to_keep),1);
    simple_ones = find(to_keep);
    small_assign(simple_ones) = 1:length(simple_ones);
    s2 = find(~to_keep);
    while ~isempty(s2)
        new_small_assign = small_assign;
        for i=1:numel(s2)
            simple_and_neighbor = find(small_assign~=0 & D(:,s2(i)) ~=0);
            idx = knnsearch(double(masks(simple_and_neighbor,:)), double(masks(s2(i),:)));
            if isempty(idx)
                continue;
            end
            new_small_assign(s2(i)) = small_assign(simple_and_neighbor(idx));
        end
        s2 = find(new_small_assign == 0);
        small_assign = new_small_assign;
%         idx = knnsearch(double(masks(simple_ones,:)),double(masks(s2,:)));
%         small_assign(s2) = idx;
    end
end

function [bndry_pairs, edgelet_ids, edgelet_sp] = ...
    generate_sp_neighborhood(sp_seg)
% create offset images to find superpixel boundary pixels
south_seg = [sp_seg(2:end,:); sp_seg(end,:)];
east_seg = [sp_seg(:,2:end), sp_seg(:,end)];

% find the actual boundary pixels
south_bndry = south_seg ~= sp_seg;
east_bndry = east_seg ~= sp_seg;

% find the src and the dst superpixel for each boundary pixel
south_sp_src = sp_seg(south_bndry);
south_sp_dst = south_seg(south_bndry);
east_sp_src = sp_seg(east_bndry);
east_sp_dst = east_seg(east_bndry);

south_bndry_pairs = find(south_bndry);
east_bndry_pairs = find(east_bndry);
bndry_pairs = [south_bndry_pairs, south_bndry_pairs+1;
    east_bndry_pairs, ...
    east_bndry_pairs+size(sp_seg,1)];

src_dst_pairs = [south_sp_src south_sp_dst; ...
    east_sp_src east_sp_dst];

[src_dst_pairs, sorted_ind] = sort(src_dst_pairs, 2);

sorted_ind = sub2ind(size(src_dst_pairs), ...
    repmat([1:size(sorted_ind,1)]',1,2), sorted_ind);

bndry_pairs = bndry_pairs(sorted_ind);

[edgelet_sp, ~, edgelet_ids] = unique(src_dst_pairs, 'rows');
end