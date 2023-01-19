function [f,g_this, g_next] = get_pairwise_loss(pcts_this, pcts_next, pairwise_link, internal_next, weights_unary)
        % First, find out what is the weight of each term
%        pairwise_link = bsxfun(@times, sqrt(weight), pairwise_link);
%        sqrt(weight)
        % While weights_unary are all in a single term, weights2_unary
        % are in multiple different terms, therefore, one need to get
        % the square first and then sum together
        new_pairwise_link = bsxfun(@times, weights_unary, pairwise_link);
        % Compute (sum(all_the_next) * this - all_the_next).^2
        % To compute (a - xb)^2, we need to get a^2 + x^2 b^2 - 2axb
        f = - sum(sum(pcts_this' * new_pairwise_link * pcts_next));
        % This is the part from the first frame
        f = f + 0.5 * sum(sum(bsxfun(@times, weights_unary, pcts_this.^2)));
        % A final part is the internal behavior within all the next
        % superpixels (e.g. (0.3 x_1 + 0.7x_2) would create a term of 
        % 2 * 0.21 x_1 x_2, for which we need to construct a square matrix made
        % by self outer-product of [0.3 0.7]' * [0.3 0.7]
        f = f + 0.5 * sum(sum(pcts_next' * internal_next * pcts_next));
        
        % two parts for g, one is for this frame, another for next frame
        g_this = -new_pairwise_link * pcts_next + bsxfun(@times, weights_unary, pcts_this);
        g_next = -new_pairwise_link' * pcts_this + internal_next * pcts_next;
end
