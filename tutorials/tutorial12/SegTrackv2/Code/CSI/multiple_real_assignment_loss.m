function [f,g] = multiple_real_assignment_loss(pcts, sup_size, sup_assignment, scores, hacks, lambda2)
    pcts = reshape(pcts, length(sup_size), size(scores,2));
    [s,~,s_others] = compute_empirical_score(pcts, sup_size, sup_assignment);
    s0 = sum(sup_assignment,2);
    
    % Segment score consistency
    f = sum(sum(bsxfun(@times, hacks, ( s - scores).^2)));
    sqrt_size_pcts = bsxfun(@times, sup_size, pcts);
    % Don't constrain the big one
    sqrt_size_pcts(end,:) = 0;
    
    % Re-scale lambda to account for the image size
    lambda2_use = lambda2 / sum(sqrt(sup_size(s0~=0)));
    % Regularization to prevent small pieces get too large weight
    f = f + lambda2_use * (pcts(:)' * sqrt_size_pcts(:));
 
% Overlap constraint    
%    for i=1:size(scores,2)
%        for j=i+1:size(scores,2)
%            f = f + lambda2 * sum(pcts(:,i) .* pcts(:,j));
%        end
%    end
    % Regularization to prevent pieces that appeared few times get too
    % large weight
%    f = f + lambda2 * sum(sum(bsxfun(@times, exp(-beta * s0), pcts)));
    if nargout > 1
    g1 = bsxfun(@times, sup_assignment, sup_size);
%    g1 = bsxfun(@rdivide, g1, s_others');
    g2 = bsxfun(@times, ~sup_assignment, sup_size);
%    g2 = bsxfun(@rdivide, g2, s_others');
    g = 2 * g1 * (bsxfun(@times,hacks, ((s - scores) ./ s_others)));
    g = g - 2 * g2 * (s .* bsxfun(@times, hacks, ((s - scores) ./ s_others)));
%    g = g / size(scores,1);
    g = g + 2 * lambda2_use * sqrt_size_pcts;
 
% Overlap constraint
%    for i=1:size(scores,2)
%        g(:,i) = g(:,i) + lambda2 * sum(pcts(:,[1:i-1 i+1:end]),2);
%    end
%    g = g + lambda2 * repmat(exp(-beta * s0),1,size(scores,2));
    if sum(isnan(g))
        disp('Please check for errors if there is NaN in g!');
        g = zeros(sum(size(pcts)),1);
    end
    g = g(:);
    end
end