function best_seg = run_cross_frame_assignment_fb(scores, sup_size, sup_assignment, pairwise_links, backward_links, Pixels, lambda, lambda2, slack)
    seg_sizes = cell(length(sup_assignment),1);
    % Relax only use segments that are big enough (all segments
    % will be used in the true objective function)
    for i=1:length(sup_assignment)
        seg_sizes{i} = sup_assignment{i}' * sup_size{i};
    end

    best_seg = cell(length(sup_assignment),size(scores{1},2));
    t = tic();
    % This only needs to be computed once and was making the code extremely
    % slow if it's computed every iteration! Just do this once.
    internal_next = cell(length(pairwise_links),1);
    internal_this = cell(length(backward_links),1);    
    sup_normalize1 = zeros(numel(sup_size),1);
% sup_normalize should be the median of occupied superpixels only to heavily penalize mapping entirely to background!    
    for i=1:numel(sup_size)
        sup_normalize1(i) = median(sup_size{i}(sum(sup_assignment{i},2)~=0));
    end
    sup_normalize = mean(sup_normalize1);
%    weights_unary = double(full(sum(pairwise_link,2))) .* sqrt(weight) .* sqrt(sum(bsxfun(@times, pairwise_link, weight2'),2));
    if lambda2 > 0
    for k=1:length(pairwise_links)
        internal_next{k} = zeros(size(pairwise_links{k},2));
        internal_this{k} = zeros(size(backward_links{k},2));
        % NOTE: weights_unary and weights2_unary needs to be consistent
        % with the ones in cross_frame_real_loss_fb
        [weights_unary2,weights_unary, bg_ws] = decide_hack_weight(pairwise_links{k}, sup_size{k}, sup_size{k+1}, sup_normalize, sup_assignment{k+1});
        % Do a bit stupid stuff of doing this more times
        if k > 1
            [~,~, bg_ws2] = decide_hack_weight(backward_links{k-1}, sup_size{k}, sup_size{k-1}, sup_normalize, sup_assignment{k-1});
            bg_ws = intersect(bg_ws, bg_ws2);
        end
        % swap 
        weights_unary(bg_ws) = weights_unary2(bg_ws);
        % bg_ws2 is the backlink from previous frame, that represents the
        % link at this frame
        [weights2_unary2,weights2_unary, bg_ws] = decide_hack_weight(backward_links{k}, sup_size{k+1}, sup_size{k}, sup_normalize, sup_assignment{k});
        if k<length(pairwise_links)-1
            [~,~, bg_ws2] = decide_hack_weight(pairwise_links{k+1}, sup_size{k+1}, sup_size{k+2}, sup_normalize, sup_assignment{k+2});
            bg_ws = intersect(bg_ws, bg_ws2);
        end
        weights2_unary(bg_ws) = weights2_unary2(bg_ws);
        for j=1:size(pairwise_links{k},1)
            internal_next{k} = internal_next{k} + weights_unary(j) * pairwise_links{k}(j,:)' * pairwise_links{k}(j,:);
        end
        for j=1:size(backward_links{k},1)
            internal_this{k} = internal_this{k} + weights2_unary(j) * backward_links{k}(j,:)' * backward_links{k}(j,:);
%            internal_next{k} = internal_next{k} +  pairwise_links{k}(j,:)' * pairwise_links{k}(j,:);
        end
    end
    end
    for j=1:size(scores{1},2)
        j
        new_scores = cellfun(@(x) x(:,j), scores, 'UniformOutput',false);
        loss_func_real = @(x) cross_frame_real_loss_fb(x,sup_size, sup_assignment,new_scores, pairwise_links,backward_links, internal_next, internal_this, lambda, lambda2);
        % Setup the optimization
        options.maxIter = 1000;
        options.verbose = 0;
        options.method = 'lbfgs';
        pcts = ones(sum(cellfun(@length,sup_size)),1);
        % Instead of 1, initialize pcts with the best segment
        counter = 0;
        for i=1:length(sup_assignment)
            [~,b] = max(scores{i}(:,j));
            pcts(counter+1:counter+length(sup_size{i})) = sup_assignment{i}(:,b);
            counter = counter + length(sup_size{i});
        end
            % No EM, only estimate the parameters
        options.optTol = 1e-6 * length(sup_assignment);
        pcts = minConf_TMP(loss_func_real,pcts, zeros(length(pcts),1), ones(length(pcts),1), options);
        % One can use this to show the optimization results if needed.
    %      pixel_assignments = pcts_to_picture(pcts, numel(pcts),1, Pixels);
    %      figure,imshow(pixel_assignments);
        counter = 0;
        for i=1:length(sup_assignment)
                [masks, fin_score(i)] = compute_optimal_mask(pcts(counter+1:counter+length(sup_size{i})), ...
                    sup_size{i}, slack);
                best_seg{i,j} = ismember(Pixels{i},find(masks));
                    counter = counter + length(sup_size{i});
        end
        % Small hack, redo the ones with very low fin_score, reduce slack
        % on them
        counter = 0;
        for i=1:length(sup_assignment)
            if fin_score(i) < mean(fin_score)
                [masks, fin_score_new(i)] = compute_optimal_mask(pcts(counter+1:counter+length(sup_size{i})), ...
                    sup_size{i}, slack * fin_score(i) / mean(fin_score));
                best_seg{i,j} = ismember(Pixels{i},find(masks));
            end
            counter = counter + length(sup_size{i});
        end
    end
    disp('Inference time: ');
    toc(t)
end

function x = simpleproj(x)
    x(x >1) = 1;
    x(x<0) = 0;
end