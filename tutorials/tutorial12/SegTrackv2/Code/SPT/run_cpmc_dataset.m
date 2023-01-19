function run_cpmc_dataset(exp_dir, imgsetpath, dataset, mask_type, pb_folder, overwrite)
    DefaultVal('overwrite',false);
    [directories, img_names] = parse_dataset(imgsetpath, dataset);
% Compute all flows
    compute_all_flow(exp_dir, directories, img_names, overwrite);
% Compute all gb
    compute_all_boundaries(exp_dir, directories, img_names, true, overwrite);
% First run CPMC on Pb only
    if ~exist('pb_folder','var') || isempty(pb_folder)
        segm_pars.pb_folder = {[exp_dir './PB/'];[exp_dir './PB_flow/'];[exp_dir './PB_plusflow/'];};
    else
        if iscell(pb_folder)
            segm_pars.pb_folder = cellfun(@(x) [exp_dir x '/'],pb_folder,'UniformOutput',false);
        else
            segm_pars.pb_folder = {[exp_dir pb_folder '/']};
        end
    end
    if exist('mask_type','var') && ~isempty(mask_type)
        segm_pars.name = mask_type;
    else
        segm_pars.name = 'WithOpticalFlow';
    end
    for i=1:length(img_names)
        cpmc_maskonly(exp_dir,directories{i}, img_names{i},segm_pars, overwrite);
    end
end