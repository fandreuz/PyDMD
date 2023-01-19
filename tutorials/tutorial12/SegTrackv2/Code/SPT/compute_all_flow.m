function compute_all_flow(exp_dir, directories, img_names, overwrite)
    if ~exist([exp_dir 'MyOpticalFlow/'],'dir')
        mkdir([exp_dir 'MyOpticalFlow/']);
    end
    DefaultVal('overwrite',false);
    new_sequence = true;
    for i=1:(length(img_names)-1)
        % Skip the last frame of one sequence
        if ~strcmp(directories{i}, directories{i+1})
            new_sequence = true;
            continue;
        end
        if ~overwrite
            listing = dir([exp_dir 'MyOpticalFlow/' directories{i} '/' img_names{i} '_' img_names{i+1} '.mat']);
            if ~isempty(listing)
                continue;
            end
        end
        % Try to figure out the extension and read the images
        if i==1 || new_sequence == true
            ext = get_img_extension([exp_dir 'JPEGImages/' directories{i} '/'],img_names{i});
            im1 = imread([exp_dir 'JPEGImages/' directories{i} '/' img_names{i} ext]);
            new_sequence = false;
        else
            im1 = im2;
        end
        ext2 = get_img_extension([exp_dir 'JPEGImages/' directories{i} '/'], img_names{i+1});
        im2 = imread([exp_dir 'JPEGImages/' directories{i} '/' img_names{i+1} ext2]);
        % this is the core part of calling the mexed dll file for computing optical flow
        % it also returns the time that is needed for two-frame estimation
        fwflow = estimate_flow_interface(im1,im2,'classic+nl-fast');
        bwflow = estimate_flow_interface(im2,im1,'classic+nl-fast');
        % Debug code
%         colorIm = flowToColor(cat(3,fwflow_vx,fwflow_vy));
%         imshow(colorIm);
%         colorIm = flowToColor(cat(3,bwflow_vx,bwflow_vy));
%         imshow(colorIm);
        if ~exist([exp_dir 'MyOpticalFlow/' directories{i}],'dir')
            mkdir([exp_dir 'MyOpticalFlow/' directories{i}]);
        end
        fwflow_vx = single(fwflow(:,:,1));
        fwflow_vy = single(fwflow(:,:,2));
        bwflow_vx = single(bwflow(:,:,1));
        bwflow_vy = single(bwflow(:,:,2));
        save([exp_dir 'MyOpticalFlow/' directories{i} '/' img_names{i} '_' img_names{i+1} '.mat'], ...
         'fwflow_vx','bwflow_vx','fwflow_vy','bwflow_vy','-v7');
    end
end
