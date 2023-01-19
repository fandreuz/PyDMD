function [] = myViewResults(img, masks, scores)

numSegs = size(masks, 3);
numPerRow = 4;
if numSegs > 100
    numSegs = 20;
end


cols = numPerRow;
rows = ceil(numSegs/numPerRow);

figure;
for i=1:numSegs   
    curImg = img;
    curImg(:,:,2) = img(:,:,2) + double(masks(:,:,i));
    curImg(curImg>1) = 1;
    subplot(rows, cols, i), imshow(curImg), title(sprintf('%4f', scores(i)))
    axis off
end
