function mq = SegTrack_segment_quality_all(ConfigFile_or_opts, dataset, suffix, mask_type, fold, overwrite)
  % each method is saved in a separate file.
  % for each method records:
  % - for each image in the database the best fmeasure
  % mask_type is for some cases when one doesn't want the default mask type in 
  % the config file.

    if ~isstruct(ConfigFile_or_opts)
        eval(ConfigFile_or_opts);
    else
        SVMSEGMopts = ConfigFile_or_opts;
    end
    DefaultVal({'suffix','mask_type','overwrite'},{[],SVMSEGMopts.mask_type,false});
  
    [directories, img_names] = parse_dataset(SVMSEGMopts.seg.imgsetpath,dataset);
  cmap = SvmSegm_labelcolormap();
  
    % Quality for all the foreground classes, no background
    Quality_all = struct();

    t1 =  tic();
    counter = 1;
    if ~isempty(suffix)
       direct = [SVMSEGMopts.segment_quality_dir mask_type '/' SVMSEGMopts.segment_quality_type '_' suffix '/' ];
    else
        direct = [SVMSEGMopts.segment_quality_dir mask_type '/' SVMSEGMopts.segment_quality_type '/' ];
    end
    if ~exist(direct, 'dir')
        mkdir(direct);
    end
    %parfor (i=1:length(img_names), 8)
    if exist('fold','var') && ~isempty(fold)
        mx = length(img_names);
        each = floor(mx / fold(2)) + 1;
        lb = 1 + (fold(1) - 1 ) * each;
        ub = fold(1) * each;
        if ub > length(img_names)
            ub = length(img_names);
        end
    else
        lb = 1;
        ub = length(img_names);
    end
    for i=lb:ub
      i      
      t = tic();
      direct2 = [direct directories{i} '/'];
      if ~exist(direct2, 'dir')
         mkdir(direct2);
      end      
      file_to_save = [direct2 img_names{i} '.mat'];
      if exist(file_to_save,'file') && ~overwrite
          continue;
      end
      var = load([SVMSEGMopts.segment_matrice_dir mask_type '/' directories{i} '/' img_names{i} '.mat']);
      masks = var.masks;
      % Added workaround because of the tiling problem.
      if iscell(masks)
          masks = masks{1};
      end
      ct_st = counter;
      for j=1:254
      % Basically we don't know anything on test, so just don't do anything.
      if strncmp(dataset,'test',4)
          Quality_all(i).class = 1;
          Quality_all(i).image_name = img_names{i};
          Quality_all(i).object = 1;
          Quality_all(i).q = zeros(size(masks,3), 1);
          Quality_all(i).sz = squeeze(sum(sum(masks,2),1));
          Quality_all(i).object_sz = 0;
          file_to_save = [direct2 img_names{i} '.mat']
          Quality = Quality_all(i);
          save(file_to_save, 'Quality');
          continue;
      else
          try
            gtobjfile = sprintf(SVMSEGMopts.seg.instimgpath,directories{i},img_names{i});
            ext_gt = get_img_extension(gtobjfile);
            single_inst = true;
          catch
              try
                  gtobjfile = sprintf(SVMSEGMopts.seg.instimgpath,[directories{i} '/' int2str(j)],img_names{i});
                  ext_gt = get_img_extension(gtobjfile);
                  single_inst = false;
              catch
                  break;
              end
          end
          [gtim2,map] = imread([gtobjfile ext_gt]);
          if size(gtim2,3) > 1
              gtim2 = gtim2(:,:,1);
          end
          locs = gtim2<255;
      end
%      nocare_bground = (gtim ~= 0);
      % Don't consider 255 which is "don't care"
%      for j=1:254
          ground_truth_seg = (gtim2 > 224);
          locs = (gtim2 > 224 | gtim2 < 32);
          % Finished for this image
          size_gt = sum(sum(ground_truth_seg));
          if size_gt == 0
              break;
          end
          
          im1 = find(ground_truth_seg==1);
          if exist('gtim','var')
          clslbl = gtim(im1(1));
          if mean(gtim(im1)) ~= max(gtim(im1))
              disp('Inconsistent annotation!');
          end
          end
          % Output both the quality and the size of the image
          % Don't use locs if doing the pixel_difference metric (Tsai 2010)
          if strcmp(SVMSEGMopts.segment_quality_type,'pixel_difference')
              ground_truth_seg = (gtim2 > 128);
              size_gt = sum(sum(ground_truth_seg));
              [q, sz]=compute_overlaps(masks,ground_truth_seg, [], SVMSEGMopts.segment_quality_type);
          else
              [q, sz]=compute_overlaps(masks,ground_truth_seg, locs, SVMSEGMopts.segment_quality_type);
          end
          Quality_all(counter).image_name = img_names{i};
          Quality_all(counter).object = 1;
          Quality_all(counter).q = q;
          Quality_all(counter).sz = sz;
          Quality_all(counter).object_sz = size_gt;
          counter = counter + 1;
          % For pixel difference, smaller is better.
          if strcmp(SVMSEGMopts.segment_quality_type,'pixel_difference')
              mq(i,j) = min(q);
          else
            mq(i,j) = max(q);
          end
          if single_inst
              break;
          end
      end
      ct_end = counter-1;
      file_to_save = [direct2 img_names{i} '.mat']
      Quality = Quality_all(ct_st:ct_end);
      save(file_to_save, 'Quality');
      end
      toc(t)
%    end 
%    if ~exist('jumped_flag','var')
%    Quality = Quality_all;
    toc(t1)
%    save([direct2 dataset '.mat'], 'Quality', '-v7.3');
    if exist('mq','var')
        disp('Mean best quality: ');
        mean(mq)
    end
%    end
end
