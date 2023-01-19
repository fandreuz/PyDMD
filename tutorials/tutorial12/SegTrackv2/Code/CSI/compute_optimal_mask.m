function [masks,scores] = compute_optimal_mask(pcts, sup_size, slack)
% compute_optimal_mask: Use a greedy algorithm that is optimal if there is
% only one category. Compute pcts/(1-pcts) and sort them. Then one can prove
% that superpixels with larger ratios need to be picked first in order to obtain an optimal
% segment. Superpixels are stopped getting picked when their ratio is
% smaller than the current score (i.e. adding it doesn't help the score).
% Because of the previous regularization tends to make numbers smaller than
% what they should be, here we allow a "slack", where we only stop
% including superpixels when pcts/(1-pcts) * (1 + slack) < scores
    pcts = reshape(pcts, length(sup_size), numel(pcts) / length(sup_size));
    gt_pixs = bsxfun(@times, pcts,sup_size);
    non_gt = bsxfun(@times, 1-pcts,sup_size);
    all_gt = sum(gt_pixs);
    ratios = pcts ./ (1-pcts);
    masks = false(size(pcts));
    scores = zeros(1,size(pcts,2));
    optimal_gts = zeros(size(scores));
    optimal_nongts = zeros(size(scores));
    for j=1:size(pcts,2)
        [sorted_ratios, plcs] = sort(ratios(:,j),'descend');
        runsum_curgt = cumsum(gt_pixs(plcs,j));
        runsum_others = cumsum(non_gt(plcs,j)) + all_gt(j);
        score_cur = runsum_curgt ./ runsum_others;
        cutoff = find(score_cur > sorted_ratios * (1 + slack), 1, 'first');
        if cutoff > 1
            masks(plcs(1:cutoff-1),j) = true;
            [scores(j),pos] = max(score_cur);
            optimal_gts(j) = runsum_curgt(pos);
            optimal_nongts(j) = runsum_others(pos);
        end
    end
    % Mutual exclusion: only one of the masks can be 1
    a = sum(masks,2);
    need_to_resolve = find(a > 1);
    need_to_resolve = need_to_resolve(end:-1:1);
    for i=1:length(need_to_resolve)
        all_cls = find(masks(need_to_resolve(i),:));

        % Loss if this superpixel get dropped, in all classes
        loss = scores - (optimal_gts - gt_pixs(need_to_resolve(i),:)) ./ (optimal_nongts - non_gt(need_to_resolve(i),:));
        [~,b] = max(loss(all_cls));
        masks(need_to_resolve(i),:) = 0;
        masks(need_to_resolve(i),all_cls(b)) = 1;
    end
    % After assignment, recompute scores
    for j=1:size(pcts,2)
        s1 = find(masks(:,j));
        if isempty(s1)
            continue;
        end
        [sorted_ratios, plcs] = sort(ratios(s1,j),'descend');
        runsum_curgt = cumsum(gt_pixs(s1(plcs),j));
        runsum_others = cumsum(non_gt(s1(plcs),j)) + all_gt(j);
        score_cur = runsum_curgt ./ runsum_others;
        scores(j) = max(score_cur);
    end    
end