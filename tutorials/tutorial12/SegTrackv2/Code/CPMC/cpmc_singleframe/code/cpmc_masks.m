% Copyright (C) 2010 Joao Carreira
%
% This code is part of the extended implementation of the paper:
% 
% J. Carreira, C. Sminchisescu, Constrained Parametric Min-Cuts for Automatic Object Segmentation, IEEE CVPR 2010
% 

function [masks] = cpmc_masks(exp_dir, directory, img_name, segm_pars, overwrite)    
    DefaultVal({'segm_pars','overwrite'}, {[],false});
    
    img_folder = [exp_dir 'JPEGImages/' directory '/'];   
    
    % extract initial pool
    dir_name = [exp_dir 'MySegmentsMat/' segm_pars.name];
    if(~exist([exp_dir 'MySegmentsMat/' segm_pars.name], 'dir'))
        mkdir(dir_name);
    end
    
    filename = [dir_name '/' directory '/' img_name '.mat'];
    if ~exist([dir_name '/' directory '/'],'dir')
        mkdir([dir_name '/' directory '/']);
    end
    if ~isempty(directory)
        segm_pars.pb_folder = cellfun(@(x) [x directory '/'], segm_pars.pb_folder, 'UniformOutput', false);
    end
    if(~exist(filename, 'file') || overwrite)
        [masks] = SvmSegm_extract_segments_nosave(img_folder, {img_name}, segm_pars);
        masks = masks{1};
        save(filename, 'masks');
    else
        var = load(filename);
        masks = var.masks;
    end
end