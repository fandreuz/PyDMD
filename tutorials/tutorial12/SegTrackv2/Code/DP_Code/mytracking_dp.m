function [inds_all, min_cs] = mytracking_dp(K, overlap_mat, c_en, c_ex, betta, thr_cost, max_it)
% Call the dynamic programming to track an object with pairwise terms in
% the cell array K and segment overlap matrices in overlap_mat.
% For the other parameters please check the original code in 
% Hamed Pirsiavash, Deva Ramanan, Charless Fowlkes, "Globally-Optimal 
% Greedy Algorithms for Tracking a Variable Number of Objects," Computer 
% vision and Pattern Recognition CVPR 2011
DefaultVal({'c_en','c_ex','max_it','thr_cost','betta'},{10,10,1e6,10,0.2});

thr_nms = 0.5;
lens = cellfun(@(x) size(x,1), K);
% Final frame
lens = [lens;size(K{end},2)];
dnum = sum(lens);
plc = [0;cumsum(lens)];
c = betta * ones(dnum,1);

dp_c     = [];
dp_link  = [];
orig     = [];
min_c     = -inf;
it        = 0;
k         = 0;
inds_all  = zeros(1,1e6);
id_s      = zeros(1,1e6);
redo_nodes = [1:dnum]';
while (min_c < thr_cost) && (it < max_it)
  it = it+1;
  if mod(it,100)==0
      it
  end
  
  dp_c(redo_nodes,1) = betta + c_en;
  dp_link(redo_nodes,1) = 0;
  orig(redo_nodes,1) = redo_nodes;
  
  for ii=1:length(redo_nodes)
    i = redo_nodes(ii);
    frame_i = find(i > plc,1,'last');
    if frame_i==1
        continue;
    end
    neighbors = K{frame_i-1}(:,i-plc(frame_i)) > 0;
    nei_inds = find(neighbors) + plc(frame_i-1);
    if isempty(nei_inds)
        continue;
    end
    [min_cost j] = min((-K{frame_i-1}(neighbors,i-plc(frame_i))) + c(i) + dp_c(nei_inds));
    min_link = nei_inds(j);
    if dp_c(i,1) > min_cost
      dp_c(i,1) = min_cost;
      dp_link(i,1) = min_link;
      orig(i,1) = orig(min_link);
    end
  end
  
  [min_c ind] = min(dp_c + c_ex);
  
  inds = zeros(dnum,1);
  
  k1 = 0;
  while ind~=0
    k1 = k1+1;
    inds(k1) = ind;
    ind = dp_link(ind);
  end
  inds = inds(1:k1);
  
  % Don't keep the tracks that don't go to near start
  if length(inds) > length(K) - 5
      if length(inds) < length(K)+1
        inds2 = [inds;zeros(length(K)+1-length(inds),1)];
      else
          inds2 = inds;
      end
      % If it doesn't start from the last frame
      if inds2(1) <= plc(end-1)
          start_frame = find(inds2(1) > plc,1,'last');
          inds2 = [zeros(length(K)+1-start_frame,1);inds2];
          inds2 = inds2(1:length(K)+1);
      end
      inds_all(k+1:k+length(inds2)) = inds2;
      id_s(k+1:k+length(inds2)) = it;
      k = k+length(inds2);
  end
  
  if exist('overlap_mat','var') && ~isempty(overlap_mat)
      supp_inds = [];
      for ii=1:length(inds)
          frame = find(inds(ii) > plc, 1,'last');
          supp_fram = find(overlap_mat{frame}(inds(ii) - plc(frame),:) >= 0.95);
          supp_inds = [supp_inds supp_fram + plc(frame)];
      end
%    supp_inds = nms_aggressive(dres, inds, thr_nms);
    origs = unique(orig(supp_inds));
    redo_nodes = find(ismember(orig, origs));
  else
    supp_inds = inds;
    origs = inds(end);
    redo_nodes = find(orig == origs);
  end
  redo_nodes = setdiff(redo_nodes, supp_inds);
  dp_c(supp_inds) = inf;
  c(supp_inds) = inf;

  min_cs(it) = min_c;
end
inds_all = inds_all(1:k);
inds_all = reshape(inds_all,length(K)+1,length(inds_all)/(length(K)+1));
inds_all = bsxfun(@minus,inds_all,plc(end-1:-1:1));
inds_all(inds_all<0) = 0;
%res = sub(dres, inds_all);
%res.id = id_s';

