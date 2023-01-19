% Copyright (C) 2010 Joao Carreira
%
% This code is part of the extended implementation of the paper:
% 
% J. Carreira, C. Sminchisescu, Constrained Parametric Min-Cuts for Automatic Object Segmentation, IEEE CVPR 2010
% 

function D = SvmSegm_compute_segm_feats_nofiles(exp_dir, directory, img_name, masks, feat_type)
    assert(iscell(feat_type));
    listing = dir([exp_dir '/JPEGImages/' directory '/' img_name '.*']);
    I = imread([exp_dir '/JPEGImages/' directory '/' listing(1).name]);
    for h=1:numel(feat_type)
        if(strcmp(feat_type{h}, 'SimpleSegmentFeatures'))
            f = SimpleSegmentFeatures(I, masks, [exp_dir 'PB/' directory '/' img_name '_PB']);
        elseif(strcmp(feat_type{h}, 'GestaltSegmentFeatures'))
            f =GestaltSegmentFeatures(I, masks, [exp_dir 'PB/' directory '/' img_name '_PB']);
        else
            error('no such feature type');
        end

        f = f.compute_all_feats();
        
        D{h} = f.get_feats();
    end    
end