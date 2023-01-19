function pixel_assignments = pcts_to_picture(pcts, num_superpixels, num_classes, Pixels)
    pcts = reshape(pcts, num_superpixels, num_classes);
    %        pcts = reshape(pcts, length(sup_size{i}), length(possible_classes)+1);
    pixel_assignments = zeros(size(Pixels,1),size(Pixels,2),num_classes);
    for j=1:num_classes
        if sum(pcts(1:end-1,j))==0
            continue;
        end
        Pixels2 = zeros(size(Pixels));
        s2 = find(pcts(1:end-1,j)~=0);
        for k=1:length(s2)
            Pixels2(Pixels==s2(k)) = pcts(s2(k),j);
        end
        pixel_assignments(:,:,j) = Pixels2;
    end
end