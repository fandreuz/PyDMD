function compute_all_boundaries(exp_dir, directories, img_names, add_flow, overwrite)
    DefaultVal({'add_flow','overwrite'},{false,false});
    for i=1:length(img_names)
        if ~exist([exp_dir 'PB/' directories{i}],'dir')
            mkdir([exp_dir 'PB/' directories{i}]);
        end
        if ~overwrite
            l1 = dir([exp_dir 'PB/' directories{i} '/' img_names{i}  '_PB.mat']);
            if add_flow
                l2 = dir([exp_dir 'PB_plusflow/' directories{i} '/' img_names{i}  '_PB.mat']);
                l3 = dir([exp_dir 'PB_flow/' directories{i} '/' img_names{i}  '_PB.mat']);
                if ~isempty(l1) && ~isempty(l2) && ~isempty(l3)
                    continue;
                end
            else
                if ~isempty(l1)
                    continue;
                end
            end
        end
        ext = get_img_extension([exp_dir 'JPEGImages/' directories{i} '/'], img_names{i});
        I = imread([exp_dir 'JPEGImages/' directories{i} '/' img_names{i} ext]);
        [gPb_thin, gPb_fat, textons] = simpleGPb(I);
        save([exp_dir 'PB/' directories{i} '/' img_names{i}  '_PB.mat'], 'gPb_thin', 'gPb_fat', 'textons');
        if add_flow
            lists = dir([exp_dir 'MyOpticalFlow/' directories{i} '/' img_names{i} '_*.mat']);
            if ~isempty(lists)
                fname = lists(1).name;
                var = load([exp_dir 'MyOpticalFlow/' directories{i} '/' fname]);
                flow_vx = var.fwflow_vx;
                flow_vy = var.fwflow_vy;
%                flow_vx = cat(3,var.fwflow_vx, - var.bwflow_vx);
%                flow_vy = cat(3,var.fwflow_vy, - var.bwflow_vy);
            end
            % If forward is not available, use backward
            lists = dir([exp_dir 'MyOpticalFlow/' directories{i} '/*_' img_names{i} '.mat']);
            if ~isempty(lists)
                fname = lists(1).name;
                var = load([exp_dir 'MyOpticalFlow/' directories{i} '/' fname]);
                flow_vx2 = var.bwflow_vx;
                flow_vy2 = var.bwflow_vy;
%                flow_vx2 = cat(3,var.bwflow_vx, - var.fwflow_vx);
%                flow_vy2 = cat(3,var.bwflow_vy, - var.fwflow_vy);
            % Let me actually add all the flows together!
            end
            if exist('flow_vx','var') && exist('flow_vx2','var')
                flow_vx = cat(3,flow_vx, flow_vx2);
                flow_vy = cat(3,flow_vy, flow_vy2);
            end
            if ~exist('flow_vx','var') && exist('flow_vx2','var')
                flow_vx = flow_vx2;
                flow_vy = flow_vy2;
            end
            % No gb_geom, that's just a hack that doesn't work
%            [~,gb_thin, gb_fat] = Gb_CSGF(I, flow_vx, flow_vy);
            [gb_thin, gb_fat] = Gb_F(I, flow_vx, flow_vy);
            gb_thin = uint8(round(255*gb_thin));
            gb_fat = gb_fat * 255;
            gPb_thin = gPb_thin / 2 + gb_thin / 2;
            gPb_fat = gPb_fat / 2 + gb_fat / 2;
            
            if ~exist([exp_dir 'PB_plusflow/' directories{i}],'dir')
                mkdir([exp_dir 'PB_plusflow/' directories{i}]);
            end
            if ~exist([exp_dir 'PB_flow/' directories{i}],'dir')
                mkdir([exp_dir 'PB_flow/' directories{i}]);
            end
            save([exp_dir 'PB_plusflow/' directories{i} '/' img_names{i}  '_PB.mat'], 'gPb_thin', 'gPb_fat', 'textons');
            gPb_thin = gb_thin;
            gPb_fat = gb_fat;
            save([exp_dir 'PB_flow/' directories{i} '/' img_names{i}  '_PB.mat'], 'gPb_thin', 'gPb_fat', 'textons');
            clear flow_vx flow_vy flow_vx2 flow_vy2
        end
    end
end
