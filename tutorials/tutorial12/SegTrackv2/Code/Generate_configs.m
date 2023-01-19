function SVMSEGMopts = Generate_configs(exp_dir, num_threads)
% Parameters: exp_dir: the root directory for storing all the data files
%             num_threads: the number of threads to be used
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Machine-depdendent configurations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Base folder for all data files
% Please put the image files into the folder
% SVMSEGMopts.real_exp_dir/JPEGImages/Sequence_Name/
% e.g. E:/SegTrack/JPEGImages/Seq1/seq1_00.jpg, seq1_01.jpg, etc.
%
DefaultVal('num_threads','12');
SVMSEGMopts.real_exp_dir = exp_dir;
% Desktop or cluster (doesn't matter for now)
SVMSEGMopts.run_place = 'desktop';
% Number of threads to use
setenv('OMP_NUM_THREADS',num_threads);

%SVMSEGMopts.pars.cache_size = 4000;
%SVMSEGMopts.pars.max_kernel_matrix_size = 10000;
%SVMSEGMopts.pars.max_memory = 4000;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Input and Output
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SVMSEGMopts.exp_dir = SVMSEGMopts.real_exp_dir;
SVMSEGMopts.segment_quality_type = 'overlap';
% Mask type should be the one used for training.
SVMSEGMopts.mask_type = 'WithOpticalFlow';
SVMSEGMopts.mask_type_test = 'WithOpticalFlow';
SVMSEGMopts.tracking_features = {'bow_dense_color_sift_3_scales_figure_300',...
                                 'bow_dense_color_sift_3_scales_ground_300'};
SVMSEGMopts.scaling_types = {'norm_1','norm_1'};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Folders (Usually don't need to be changed). Changes can lead 
%% to errors since Joao did not adhere to usage of these variables.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SVMSEGMopts.results_folder = [SVMSEGMopts.exp_dir 'MyResults/'];
SVMSEGMopts.segment_quality_dir = [SVMSEGMopts.exp_dir 'SegmentEval/'];
%segment_quality_dir = SVMSEGMopts.segment_quality_dir ;
SVMSEGMopts.attention_models_dir = [SVMSEGMopts.exp_dir 'MyAttentionModels/'];
SVMSEGMopts.labeling_ranker_model_dir = [SVMSEGMopts.exp_dir 'MyLabelingModels/'];
%attention_models_dir =SVMSEGMopts.attention_models_dir ;
SVMSEGMopts.dmatrix_folder = [SVMSEGMopts.exp_dir 'MyDistanceMatrices/'];     
SVMSEGMopts.classifier_folder = [SVMSEGMopts.exp_dir 'MyClassifiers/'];
SVMSEGMopts.segment_folder = [SVMSEGMopts.exp_dir 'MyFilteredSegments/'];
SVMSEGMopts.kernel_folder = [SVMSEGMopts.exp_dir 'MyDistanceMatrices/'];
SVMSEGMopts.feats_dir = [SVMSEGMopts.exp_dir 'MyFeatures/'];
SVMSEGMopts.measurement_dir = [SVMSEGMopts.exp_dir 'MyMeasurements/'];
SVMSEGMopts.overlaps_dir = [SVMSEGMopts.exp_dir 'MyOverlaps/'];
SVMSEGMopts.segment_matrice_dir = [SVMSEGMopts.exp_dir 'MySegmentsMat/'];
SVMSEGMopts.segment_result_dir = [SVMSEGMopts.exp_dir '/../results/VOC2011/Segmentation/Carreira'];
SVMSEGMopts.superpixel_dir = [SVMSEGMopts.exp_dir 'MySuperPixels/'];
SVMSEGMopts.fourier_folder = [SVMSEGMopts.exp_dir 'MyFourierFeatures/'];
SVMSEGMopts.opticalflow_dir = [SVMSEGMopts.exp_dir 'MyOpticalFlow/'];
% Somehow imgsetpath and seg.imgsetpath must both be there and be the same
% use det.imgsetpath to denote the other one
SVMSEGMopts.imgsetpath = [SVMSEGMopts.exp_dir 'ImageSets/%s.txt'];
SVMSEGMopts.seg.imgsetpath = SVMSEGMopts.imgsetpath;
SVMSEGMopts.seg.instimgpath = [SVMSEGMopts.exp_dir 'GroundTruth/%s/%s'];
