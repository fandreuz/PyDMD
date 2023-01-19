function [pred, w, LinReg_objs] = do_traintest_1frame(train_data, test_data, train_y, lambda, rf_obj, LinReg_objs, seqs, weights)

X1 = rf_featurize(rf_obj, train_data{1}');
%[X1,pc1] = rf_pca_featurize(rf_obj, all_featframe1(1:300,:)',true, 500);
X2 = rf_featurize(rf_obj, train_data{2}');
%[X2,pc2] = rf_pca_featurize(rf_obj, all_featframe1(1:300,:)',true, 500);
allX = [X1 X2];
if ~exist('LinReg_objs','var') || isempty(LinReg_objs)
    LinReg_objs{1} = LinearRegressor_Data(allX,train_y);
else
    % Everybody adds the same
    if ~exist('seqs','var')
        tempobj = LinearRegressor_Data(allX, train_y);
        for i=1:length(LinReg_objs)
            if LinReg_objs{i}.no_target()
                continue;
            end
            if ~isempty(LinReg_objs{i})
                LinReg_objs{i} = LinReg_objs{i} + tempobj;
            else
                LinReg_objs{i} = tempobj;
            end
        end
    else
    % The nd^2 Hessian additive factor only has to be computed once, this 
    % is some computation we can save! Note it doesn't work with weighted examples.
    if ~exist('weights','var') || isempty(weights)
    Hess_Addition = allX' * allX;
    end
    for i=1:length(LinReg_objs)
        if ~isempty(LinReg_objs{i})
            if LinReg_objs{i}.no_target()
                continue;
            end
            if ~exist('weights','var') || isempty(weights)
                LinReg_objs{i} = LinReg_objs{i} + LinearRegressor_Data(allX, train_y(:,seqs{i}),[],Hess_Addition);
            else
                LinReg_objs{i} = LinReg_objs{i} + LinearRegressor_Data(allX, train_y(:,seqs{i}),weights(:,i));
            end
        else
            if ~exist('weights','var') || isempty(weights)
                LinReg_objs{i} = LinearRegressor_Data(allX, train_y(:,seqs{i}), [], Hess_Addition);
            else
                LinReg_objs{i} = LinearRegressor_Data(allX, train_y(:,seqs{i}), weights(:,i));
            end
        end
    end
    end
end
XX1 = rf_featurize(rf_obj, test_data{1}');
XX2 = rf_featurize(rf_obj, test_data{2}');
%XX1 = rf_pca_featurize(pc1, all_featframe2(1:300,:)');
%XX2 = rf_pca_featurize(pc2, all_featframe2(301:600,:)');
allX_test = [XX1 XX2];
pred = cell(length(LinReg_objs),1);
w = cell(length(LinReg_objs),1);
t = tic();
for i=1:length(LinReg_objs)
    if LinReg_objs{i}.no_target()
        continue;
    end
    w{i} = LinReg_objs{i}.Regress(lambda);
    t2 = tic();
%     pred{i} = zeros(size(allX_test,1),size(w{i},2));
%     for j=1:size(w{i},2)
%         pred{i}(overlay_mat(:,j),j) = w{i}(1,j) + allX_test(overlay_mat(:,j),:) * w{i}(2:end,j);
%     end
    pred{i} = bsxfun(@plus, w{i}(1,:), allX_test * w{i}(2:end,:));
    disp(['Testing time ' int2str(i)]);
    toc(t2);
end
disp('Regress+Testing time: ')
toc(t);
end