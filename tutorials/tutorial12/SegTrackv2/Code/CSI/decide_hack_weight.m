function [weights_unary, weights_orig, bg_ws] = decide_hack_weight(pairwise_links, sup_size_this, sup_size_next, sup_normalize, sup_assignment_next)
        weights_unary = double(full(sum(pairwise_links,2))) .* sup_size_this ./ sup_normalize;
        [bg_ws, bg_idx] = find_only_connect_bg(pairwise_links, sup_assignment_next);
        weights_orig = weights_unary;
        weights_unary(bg_ws) = max(sup_size_next(bg_idx)) ./ sup_normalize;
end

function [bg_ws, bg_idx] = find_only_connect_bg(pairwise_links, sup_assignment_next)
    all_bg = sum(sup_assignment_next,2) == 0;
    bg_idx = find(all_bg);
    % Find the superpixels that are only connected to background
    bg_ws = find(max(pairwise_links(:,all_bg),[],2));
    % Verify that they are indeed not connected to other foreground
    % superpixels
    indeed_bg = sum(pairwise_links(bg_ws, ~all_bg),2) == 0;
    bg_ws = bg_ws(indeed_bg);
end