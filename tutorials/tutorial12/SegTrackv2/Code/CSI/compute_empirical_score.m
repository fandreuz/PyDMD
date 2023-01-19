function [s,s2,s_others,ss] = compute_empirical_score(pcts, sup_size, sup_assignment)
    pcts = reshape(pcts, length(sup_size), numel(pcts) / length(sup_size));
    gt_pixs = bsxfun(@times, pcts,sup_size);
    s1 = sup_assignment' * gt_pixs;
    s2 = sup_assignment' * sup_size + 1e-10;
    s_others = bsxfun(@plus, bsxfun(@plus, -s1, sum(gt_pixs,1)), s2) + 1e-10;
    s = bsxfun(@rdivide,s1,s_others);
    if nargout > 3
        ss = bsxfun(@rdivide,s1,s2);
    end
end