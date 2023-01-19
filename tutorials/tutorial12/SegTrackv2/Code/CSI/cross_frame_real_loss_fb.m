function [f,g] = cross_frame_real_loss_fb(pcts, sup_sizes, sup_assignments, scores, pairwise_links, backward_links, internal_next, internal_this, ...
    lambda, lambda2)
% Currently don't support yet using multiple columns of scores!
    f = 0;
    g = zeros(length(pcts),1);
    counter = 0;
    sup_normalize1 = zeros(numel(sup_sizes),1);
% sup_normalize should be the median of occupied superpixels only to heavily penalize mapping entirely to background!    
    for i=1:numel(sup_sizes)
        sup_normalize1(i) = median(sup_sizes{i}(sum(sup_assignments{i},2)~=0));
    end
    sup_normalize = mean(sup_normalize1);
%    sup_normalize = mean(cellfun(@median, sup_sizes));
    for i=1:length(sup_sizes)
        this_len = length(sup_sizes{i})*size(scores{1},2);
        hack_weight = ones(size(scores{i},1),1);
        if i==1
            pcts_this = pcts(counter+1:counter+ this_len);
        end
        [f0,g0] = multiple_real_assignment_loss(pcts_this(:), sup_sizes{i}, sup_assignments{i}, double(scores{i}),hack_weight,lambda);
       f = f+f0;
       g(counter+1:counter+this_len) = g(counter+1:counter+this_len) + g0;
       % Difference is that we will add the cross-frame specified by
       % pairwise_links    
       if i<length(sup_sizes) && lambda2 > 0
           next_counter = counter + this_len;
           next_len = length(sup_sizes{i+1})*size(scores{1},2);
            pcts_next = pcts(next_counter+1:next_counter+next_len);
            pcts_this = reshape(pcts_this, length(sup_sizes{i}), size(scores{1},2));
            pcts_next = reshape(pcts_next, length(sup_sizes{i+1}), size(scores{1},2));
                    
            [weights_unary2,weights_unary, bg_ws] = decide_hack_weight(pairwise_links{i}, sup_sizes{i}, sup_sizes{i+1}, sup_normalize, sup_assignments{i+1});
            % Do a bit stupid stuff of doing this more times
            if i > 1
                [~,~, bg_ws2] = decide_hack_weight(backward_links{i-1}, sup_sizes{i}, sup_sizes{i-1}, sup_normalize, sup_assignments{i-1});
                bg_ws = intersect(bg_ws, bg_ws2);
            end
            % swap 
            weights_unary(bg_ws) = weights_unary2(bg_ws);
            % bg_ws2 is the backlink from previous frame, that represents the
            % link at this frame
            [weights2_unary2,weights2_unary, bg_ws] = decide_hack_weight(backward_links{i}, sup_sizes{i+1}, sup_sizes{i}, sup_normalize, sup_assignments{i});
            if i<length(pairwise_links)-1
                [~,~, bg_ws2] = decide_hack_weight(pairwise_links{i+1}, sup_sizes{i+1}, sup_sizes{i+2}, sup_normalize, sup_assignments{i+2});
                bg_ws = intersect(bg_ws, bg_ws2);
            end
            weights2_unary(bg_ws) = weights2_unary2(bg_ws);
            
            
%            weights_unary = decide_hack_weight(pairwise_links{i}, backward_links{i-1}, sup_sizes{i}, sup_sizes{i+1}, sup_normalize, sup_assignments{i+1});
%            weights2_unary = decide_hack_weight(backward_links{i}, sup_sizes{i+1}, sup_sizes{i}, sup_normalize, sup_assignments{i});
            [f1,g1_this, g1_next] = get_pairwise_loss(pcts_this, pcts_next,pairwise_links{i}, internal_next{i}, weights_unary);
            [f2,g2_next, g2_this] = get_pairwise_loss(pcts_next, pcts_this, backward_links{i}, internal_this{i}, weights2_unary);
            f = f + lambda2 * (f1 + f2);
            g(counter+1:counter+this_len) = g(counter+1:counter+this_len) + lambda2 * (g1_this + g2_this);
            g(next_counter+1:next_counter+next_len) = g(next_counter+1:next_counter+next_len) + lambda2 * (g1_next + g2_next);
            counter = next_counter;
            pcts_this = pcts_next;
       end
    end
end