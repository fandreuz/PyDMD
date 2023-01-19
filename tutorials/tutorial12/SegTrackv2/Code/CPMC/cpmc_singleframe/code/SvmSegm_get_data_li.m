function [Feats, expanded_labels, scaling, ids_in_bag, segment_qualities, dims] = SvmSegm_get_data_li(ConfigFile_or_opts, input_feats, data, segm_pars, scaling_type, scaling, is_test_set)
 % Two new parameters: merge_qualities merge all the qualities (for different objects) in one image
 % to a single set of qualities in all the classes, segment_qualities is a cell array of size num_images*num_classes.
 % ignore_classes maximizes for all the classes, used for linear
 % regression, segment_qualities is a cell array of size num_images.
 % expanded_labels are not useful right now, since it only outputs the
 % labels of all the objects.
 %
 % dims is the ids of the dimensions of each feature type
 % is_test_set indicates that data is from the test set and we don't have
 % ground truth qualities
 
 % segm_pars: must set mask_type. All others are optional.
 %            optional: merge_qualities, ignore_classes,
 %            with_agglomerative_clustering, min_n, max_n,
 %            percentage_threshold, threshold, custom_data, quality_type,
 %            rank_quality_type
 % SVMSEGMopts: besides the directories, only need nclasses
 
     DefaultVal('is_test_set', 'false');
     if ~isstruct(ConfigFile_or_opts)
         eval(ConfigFile_or_opts)
     else
         SVMSEGMopts = ConfigFile_or_opts;
     end
     mask_type = segm_pars.mask_type;

     Feats = [];
     iFeats = {};
     if ~iscell(input_feats)
         input_feats = {input_feats};
     end
     if ~iscell(data)
         data = {data};
     end
     if isfield(segm_pars,'merge_qualities')
         merge_qualities = segm_pars.merge_qualities;
     else
         merge_qualities = false;
     end
     if isfield(segm_pars,'ignore_classes')
         ignore_classes = segm_pars.ignore_classes;
     else
         ignore_classes = false;
     end

%      % if this is the test set but you want the segments sorted by quality
%      % (which doesn't exist), then you need to pass a ranker! And we need
%      % to build a file with dummy qualities (if it doesn't exist already). That happens here.
%      if(is_test_set && segm_pars.sorted)
%          assert(isfield(segm_pars, 'rank_quality_type') && ~isempty(segm_pars.rank_quality_type));
%          if(~exist([SVMSEGMopts.exp_dir 'SegmentEval/' mask_type '/' data '.mat']))
%              img_names = textread([exp_dir 'ImageSets/Segmentation/' data '.txt'], '%s');
%              
%              % generate dummy file
%              for i=1:numel(img_names)
%              end
%              % you need to pass predicted segment score in order to get the features from the
%              % test set
%          end
%      end

  if iscell(mask_type)
      for ii=1:length(mask_type)
          if isfield(segm_pars,'custom_data') && ~isempty(segm_pars.custom_data{ii})
              cur_data = segm_pars.custom_data{ii};
          else
              cur_data = data;
          end
          for i=1:length(cur_data)
            counter = int32((ii - 1)* length(cur_data) + i);
            [iFeats{counter}, the_expanded_labels{counter}, segment_qual{counter}, the_ids_in_bag{counter}, non_vec_data, dims] = ...
            get_single_mask_data(SVMSEGMopts, input_feats, mask_type{ii},cur_data{i}, segm_pars, is_test_set);
          end
      end
  else
      if isfield(segm_pars,'custom_data') && ~isempty(segm_pars.custom_data)
          cur_data = segm_pars.custom_data;
      else
          cur_data = data;
      end
      for i=1:length(cur_data)
          [iFeats{i}, the_expanded_labels{i}, segment_qual{i}, the_ids_in_bag{i}, non_vec_data, dims] = get_single_mask_data(SVMSEGMopts, input_feats, mask_type,cur_data{i}, segm_pars, is_test_set);
      end
  end
  
  for i=1:length(iFeats)
      Feats = [Feats iFeats{i}];
      iFeats{i} = {};
  end
  ids_in_bag = [];
  for i=1:length(the_ids_in_bag)
      ids_in_bag = [ids_in_bag; the_ids_in_bag{i}];
  end
  segment_q = [];
  for i=1:length(segment_qual)
      size(segment_qual{i})
      segment_q = [segment_q;segment_qual{i}];
  end
  expanded_labels = [];
  for i=1:length(the_expanded_labels)
  	  expanded_labels = [expanded_labels the_expanded_labels{i}];
  end
    
  Feats = single(Feats);
  disp(['Get ' int2str(size(Feats,2)) ' segments successfully.']);
  counter = 1;
  if ~exist('scaling', 'var')
      scaling = cell(length(dims),1);
      input_scaling = false;
  else
      input_scaling = true;
  end
  % Do the scaling, allow different scaling on each feature
  for i=1:length(dims)
      if iscell(scaling_type)
          the_scaling_type = scaling_type{i};
      else
          the_scaling_type = scaling_type;
      end
      if(input_scaling && ~isempty(scaling) && ~any(non_vec_data))
          [Feats(counter:counter+dims(i)-1,:), scaling{i}] = scale_data(Feats(counter:counter+dims(i)-1,:),the_scaling_type,scaling{i});
      elseif ~any(non_vec_data)
          [Feats(counter:counter+dims(i)-1,:),b] = scale_data(Feats(counter:counter+dims(i)-1,:), the_scaling_type);
          scaling{i} = b;
%          [Feats(counter:counter+dims(i)-1,:), scaling{i}] = scale_data(Feats(counter:counter+dims(i)-1,:),the_scaling_type);
      else
          scaling{i} = [];
      end
      counter = counter + dims(i);
  end
  
  max_classes = max(expanded_labels);
  if ignore_classes || ~merge_qualities
      if((size(segment_q{1},2) > size(segment_q{1},1)) && (size(segment_q,1) > size(segment_q,2)))
          segment_q = segment_q';
      end      
%      segment_q
      segment_qualities = cell2mat(segment_q);
  else
      max_n = sum(cellfun(@length, ids_in_bag));
      segment_qualities = zeros(max_n, size(segment_q,2));
      counter = 1;
      for i=1:length(ids_in_bag)
        real_n = length(ids_in_bag{i});
        for j=1:max_classes
            if ~isempty(segment_q{i,j})
                segment_qualities(counter:counter+real_n-1,j)  = segment_q{i,j}(1:real_n);
            end
        end
        counter = counter + real_n;
      end
  end
end

function [iFeats, the_expanded_labels, segment_qual, the_ids_in_bag,non_vec_data, dims] = get_single_mask_data(ConfigFile_or_opts, input_feats, mask_type_in, data, segm_pars, is_test_set)
  count = 1;
    if isstruct(ConfigFile_or_opts)
        SVMSEGMopts = ConfigFile_or_opts;
    else
        eval(ConfigFile_or_opts);
    end
    mask_type = mask_type_in;
    if ~isfield(segm_pars, 'max_n')
        max_n = inf;
    else
        max_n = segm_pars.max_n;
    end
  overlap_dir = [SVMSEGMopts.real_exp_dir 'MyOverlaps/'];
    if ~isfield(segm_pars, 'min_n')
        min_n = 1;
    else
    min_n = segm_pars.min_n;
    end
    if ~isfield(segm_pars,'threshold')
        threshold = -1;
    else
        threshold  = segm_pars.threshold;
    end
    if isfield(segm_pars,'percentage_threshold')
        percentage = segm_pars.percentage_threshold;
    else
        percentage = -1;
    end
    if isfield(segm_pars,'merge_qualities')
        merge_qualities = segm_pars.merge_qualities;
    else
        merge_qualities = false;
    end
    if isfield(segm_pars,'ignore_classes')
        ignore_classes = segm_pars.ignore_classes;
    else
        ignore_classes = false;
    end
    % Get the basic informations from basic_quality_type, and other info from
    % quality_type
    if ~isfield(segm_pars,'quality_type')
        quality_type = 'overlap';
    else
        quality_type = segm_pars.quality_type;
    end
    if ~isfield(segm_pars,'sorted')
        sorted = false;
    else
        sorted = segm_pars.sorted;
    end
    img_names = textread(sprintf(SVMSEGMopts.seg.imgsetpath,data),'%s');
%    segment_quality_file = [SVMSEGMopts.segment_quality_dir '/' mask_type '/' quality_type '/' data '.mat'];

    if(isfield(segm_pars, 'rank_quality_type'))
        rank_quality_type = segm_pars.rank_quality_type;
%        segment_rank_quality_file = [SVMSEGMopts.segment_quality_dir '/' mask_type '/' rank_quality_type '/' data '.mat'];
    else
        rank_quality_type = [];
    end


    non_vec_data = false(length(input_feats),1);
    if ~isempty(input_feats)
        for j=1:length(input_feats)
        disp(['Getting data for feature: ' input_feats{j}]);
%        vars = load([SVMSEGMopts.feats_dir mask_type '/' data '__' input_feats{j}], 'Meas');

%        if(isempty(vars.Meas))
%            error(['In ' mask_type '/' data '__' input_feats{j} ', the feature is empty.']);
%        end 

%        if(iscell(vars.Meas{1}))
%            non_vec_data(j) = true;
%        end
        [jFeats{j}, the_expanded_labels, segment_qual, the_ids_in_bag, non_vec] = my_img_to_segms(SVMSEGMopts,img_names, mask_type, quality_type, rank_quality_type,input_feats{j},...
            max_n, min_n, percentage, threshold, sorted, merge_qualities, ignore_classes, overlap_dir, is_test_set);
        if non_vec
            non_vec_data(j) = true;
        end
%        [jFeats{j}, the_expanded_labels, segment_qual, the_ids_in_bag] = img_to_segms(vars.Meas, segment_quality_file, max_n, min_n, percentage, threshold, sorted, merge_qualities, segment_rank_quality_file, ignore_classes, overlap_dir, is_test_set);
        dims(j) = size(jFeats{j},1);
        clear vars;
        count = count + 1;
        end
    iFeats = zeros(sum(dims),size(jFeats{1},2));
    counter = 1;
    for j=1:length(input_feats)
        iFeats(counter:counter+dims(j)-1,:) = jFeats{j};
        counter = counter + dims(j);
    end
    clear jFeats;
        
    else
        [~, the_expanded_labels, segment_qual, the_ids_in_bag] = my_img_to_segms(SVMSEGMopts,img_names, mask_type, quality_type, rank_quality_type,[],...
            max_n, min_n, percentage, threshold, sorted, merge_qualities, ignore_classes, overlap_dir, is_test_set);
        iFeats = [];
        dims = [];
    end
    % it's hugely complicated so that it uses less memory than simply
    % calling cell2mat (for some large cases i wasn't able to)
end

% Problem: multiple objects within 1 image
function [feats, new_labels, segment_qualities, ids_in_bag, non_vec] = my_img_to_segms(SVMSEGMopts, img_names, mask_type, quality_type, rank_quality_type, input_feat,...
            max_n, min_n, percentage, threshold, sorted, merge_qualities, ignore_classes, overlap_dir, is_test_set)
%    assert(iscell(Meas));
    feature_origin_img = zeros(length(img_names),1);
    non_vec = false;
    % m is the number of objects in total, i don't know yet!
    if merge_qualities
        ids_in_bag = [];
    else
        ids_in_bag = cell(length(img_names),1);
    end
    feats = cell(length(img_names),1);
    segment_qualities = cell(length(img_names),1);
    attention_note = false;
    new_labels = [];
    
    for i=1:length(img_names)
        Q1 = [];
        if ~isempty(input_feat)
            meas_path = [SVMSEGMopts.measurement_dir mask_type '/' input_feat '/' img_names{i} '.mat'];
            try
                var = load(meas_path);
            catch e
                error(['Failed to load measurement from ' meas_path]);
            end
            Meas = var.D;
        end
        
        qual_path = [SVMSEGMopts.segment_quality_dir mask_type '/' quality_type '/' img_names{i} '.mat'];
        if ~isempty(rank_quality_type)
            rank_qual_path = [SVMSEGMopts.segment_quality_dir mask_type '/' rank_quality_type '/' img_names{i} '.mat'];
        else
            rank_qual_path = [];
        end
        try 
            var = load(qual_path);
            Quality = var.Quality;
            if iscell(Quality)
                Quality = Quality{1};
            end
            Q1 = {Quality.q};
            new_labels = [new_labels Quality.class];
            if merge_qualities
                [non_ignore,Q1] = merge_qual(Quality, 1);
                if ~ignore_classes
                    segment_qualities(i,1:length(non_ignore)) = non_ignore;
                else
                    segment_qualities{i} = Q1{1};
                end
                Q1 = Q1{1};
            else
                segment_qualities{i} = Q1;
                Q1 = max(cell2mat(Q1),[],2);
            end
        catch e
            if is_test_set
                if ~isempty(input_feat)
                    Q1 = zeros(size(Meas,2),1);
                else
                    Q1 = 0;
                end
                segment_qualities = cell(length(img_names),1);
                new_labels = uint8(1);
            else
                error(['Failed to load train quality from ' qual_path]);
            end
        end
        % max_n_vectors?
        threshold2 = -1;
        if exist('rank_qual_path','var') && ~isempty(rank_qual_path)
            if exist(rank_qual_path,'file')
                Q1 = load(rank_qual_path);
                Q1 = Q1.Quality;
            else
                if ~attention_note
                    disp(['Attention model ' rank_qual_path ' not found!']);
                    attention_note = true;
                end
            end
        end
        if isempty(Q1)
            continue;
        end
        if sorted
            [val, id] = sort(Q1,'descend');
            if percentage > 0
                threshold2 = val(1) * percentage;
            end
        else
            val = Q1;
            id = 1:length(val);
        end
        
        if (~isempty(input_feat) && length(id) ~= size(Meas,2))
            error(['In image ' img_names{i} ', number of segments in the quality file doesn''t match number of segments in the measurement file. Quality file has ' num2str(length(id)) ' while measurement file has ' num2str(size(Meas{measure_anchor},2)) ';The Quality_file is ' segment_quality_file]);
        end
        
        if threshold2 < threshold
            threshold2 = threshold;
        end
        real_n = min(length(id), max(min_n, min(sum(val>=threshold2),max_n)));
        if real_n == 0
            new_labels(i) = 0;
            continue;
        end
        if ~merge_qualities
            ids_in_bag{i} = [ids_in_bag{i};id(1:real_n)];
        else
            ids_in_bag{i} = id(1:real_n);
        end
        if ~isempty(input_feat)
            feats{i} = [feats{i};Meas(:,id(1:real_n))];
        end
        
        if ignore_classes || ~merge_qualities
            segment_qualities{i} = val(1:real_n);
            if (any(isnan(segment_qualities{i}) | isinf(segment_qualities{i})))
                disp('Nan or Inf in the quality file! Fixing them to 0, but please check for errors.');
                segment_qualities{i}(isnan(segment_qualities{i})) = 0;
                segment_qualities{i}(isinf(segment_qualities{i})) = 0;
            end
        else
            for j=1:max(new_labels)
                if ~isempty(segment_qualities{i,j})
                    segment_qualities{i,j} = segment_qualities{i,j}(id(1:real_n));
            if (any(isnan(segment_qualities{i,j}) | isinf(segment_qualities{i,j})))
                disp('Nan or Inf in the quality file! Fixing them to 0, but please check for errors.');
                segment_qualities{i,j}(isnan(segment_qualities{i,j})) = 0;
                segment_qualities{i,j}(isinf(segment_qualities{i,j})) = 0;
            end

                else
                    segment_qualities{i,j} = [];
                end
            end
        end
    end
    
%    assert(nargin >= 8);
    
%     if ~isempty(segment_rank_quality_file) && exist(segment_rank_quality_file,'file')
%         ranker = load(segment_rank_quality_file);
%         fprintf(1,'Using attention model %s\n',segment_rank_quality_file);
%     else
%     end
    feats = cell2mat(feats');
    label1 = (new_labels~=0);
    new_labels = new_labels(label1);
    if any(isnan(feats) | isinf(feats))
        disp('Find NaN or Inf in Features! Fixing them to 0, but please check for errors!');
        feats(isnan(feats)) = 0;
        feats(isinf(feats)) = 0;
    end
end

