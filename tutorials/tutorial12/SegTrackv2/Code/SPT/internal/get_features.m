
function features = get_features(feature_types, measurement_dir, scaling_types, mask_type, directory, img_name)
    features = cell(length(feature_types),1);
    for j=1:length(feature_types)
        var = load([measurement_dir mask_type '_' feature_types{j} '/' directory '/' img_name '.mat']);
        features{j} = scale_data(single(var.D), scaling_types{j});
        features{j}(isnan(features{j})) = 0;
        features{j}(isinf(features{j})) = 0;
    end
end