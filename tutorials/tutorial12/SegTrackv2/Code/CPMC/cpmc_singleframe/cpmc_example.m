clear all

%% for figure ground proposal
% replacement of gPb
addpath('./Gb_Code/');

% main code
addpath('./code/');

% other libs
addpath('./external_code/');

% para min cut
addpath('./external_code/paraFmex/');

% VL feat, used only for clustering. Someone should remove this
run('./external_code/vlfeat/toolbox/vl_setup')

% over segmentation (efficient graph cut)
addpath('./external_code/imrender/vgg/');
addpath('./external_code/immerge/');

%% for ranking
% color sift feature
addpath('./external_code/color_sift/');
% chi2 kernel
addpath('./external_code/mpi-chi2-v1_5/');

%% set up threads
% create multiple threads (set how many you have)
N_THREADS = 2;
if(matlabpool('size')~=N_THREADS)
  matlabpool('open', N_THREADS);
end

%% run cpmc
exp_dir = './data/';
%img_name = 'test'; % airplane and people
%img_name = '2007_009084'; % dogs, motorbike, chairs, people
%img_name = '2010_002868'; % buses
img_name = '2010_003781'; % cat, bottle, potted plants

tic
[masks, scores] = cpmc(exp_dir, img_name);
toc


%% show the results
I = imread([exp_dir '/JPEGImages/' img_name '.jpg']);
myViewResults(im2double(I), masks, scores)


% % visualization and ground truth score for whole pool
% %     fprintf(['Best segments from initial pool of ' int2str(size(masks,3))]);
% %     Q = SvmSegm_segment_quality(img_name, exp_dir, masks, 'overlap');
% %     save('duh_32.mat', 'Q');
% %     avg_best_overlap = mean(max([Q.q]))
% %     SvmSegm_show_best_segments(I,Q,masks);
% %
% % visualization and ground truth score for top 200 segments
% top_masks = masks(:,:,1:min(200, size(masks,3)));
% figure;
% disp('Best 200 segments after filtering');
% Q = SvmSegm_segment_quality(img_name, exp_dir, top_masks, 'overlap');
% %avg_best_overlap = mean(max([Q.q]))
% SvmSegm_show_best_segments(I,Q,top_masks);
% fprintf('Best among top 200 after filtering\n\n');
% scores(1:200)

matlabpool('close');
