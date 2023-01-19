function new_mask = gen_mask_in_new_frame(mask, flow_vec)
    if size(mask,1) ~= size(flow_vec,1) || size(mask,2) ~= size(flow_vec,2)
        error('Inconsistent size between the mask and the optical flow!');
    end
    if size(flow_vec,3) ~= 2
        error('Optical flow should be size x*y*2!');
    end
    [row,col] = find(mask);
    inds = sub2ind(size(mask), row, col);
    flow_s = reshape(flow_vec,size(flow_vec,1) * size(flow_vec,2), size(flow_vec,3));
    offset = flow_s(inds,:);
    new_locs = uint32([row col] + offset);
    still_in_image = new_locs(:,1) > 0 & new_locs(:,2) > 0 & new_locs(:,1) <= size(mask,1) & new_locs(:,2) <= size(mask,2);
    new_locs = new_locs(still_in_image,:);
    new_mask = false(size(mask));
    new_locs_ind = sub2ind(size(mask), new_locs(:,1), new_locs(:,2));
    new_mask(new_locs_ind) = true;
end