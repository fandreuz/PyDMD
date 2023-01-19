function [f,sz] = compute_overlaps(masks, GT, care_part, type)
    gtcen = regionprops(GT,'Centroid');
    f = zeros(size(masks,3),1);
    % This needs memory though, decide later if it's worth it
%    masks = uint8(masks);
    masks_pl_gt = bsxfun(@plus, uint8(GT) * 2, uint8(masks));
    masks_pl_gt = reshape(masks_pl_gt, size(masks_pl_gt,1) * size(masks_pl_gt,2), size(masks_pl_gt,3));
    if exist('care_part','var') && ~isempty(care_part)
        all_is = histc(masks_pl_gt(care_part,:), 0:3,1);
    else
        all_is = histc(masks_pl_gt, 0:3,1);
    end
    all_is = all_is';
    if ~exist('type','var') || isempty(type) || strcmp(type,'overlap')
        f = all_is(:,4) ./ (sum(all_is(:,2:4),2) + eps);
    elseif strcmp(type,'intersect_gt')
        f = all_is(:,4) ./ ((all_is(:,2) + all_is(:,4))+eps);
    elseif strcmp(type,'centroid_displacement')
        for i=1:size(masks,3)
            maskcen = regionprops(masks(:,:,i),'Centroid');
            gtcen = gtcen.Centroid;
            maskcen = maskcen.Centroid;
            f(i) = gtcen - maskcen;
        end
    elseif strcmp(type,'pixel_difference')
        f = all_is(:,2) + all_is(:,3);
    elseif strcmp(type,'pixel_percentage')
        f = (all_is(:,4) + all_is(:,1)) ./ (sum(all_is,2) + eps);
    end
    if nargout > 1
        sz = squeeze(sum(sum(masks,1),1));
    end
end