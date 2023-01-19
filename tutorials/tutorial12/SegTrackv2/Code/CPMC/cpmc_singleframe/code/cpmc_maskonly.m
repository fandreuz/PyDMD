% function [masks, scores] = cpmc(exp_dir, img_name, diversify_const, ranker_file, segm_pars)
%
% Implementation of the CPMC algorithm. It computes a ranked set of figure-ground segmentation, with associated scores.
% 
% Inputs: 
%   - exp_dir is the folder where the data is (images, ranker models,
%   codebooks, etc.)
%   - img_name is the name of the image without the ending, assuming a jpg file
%   - diversify_const is the the diversification parameter (between 0 and 1).
%       zero will give a random ranking, one will be a very redundant
%       ranking. Default value is 0.75.
%   - ranker_file is the filename of the ranker model (defaults to the
%   version trained on the training set of the VOC2010 segmentation
%   dataset). Please switch to the version trained on trainval for testing 
%   on imgs not in the train/validation sets.
%   - segm_pars is a struct with default value defined below.
%   Check the code for details.
%
% Outputs: 
%   - masks: the figure-ground segmentations in logical matrices, sorted by score
%   - scores: scores for the sorted list of masks
% 
% If you use this code please cite the following references:
%
% @inproceedings{carreira_cvpr10,
%  author = {J. Carreira and C. Sminchisescu},
%  title = {{Constrained Parametric Min-Cuts for Automatic Object Segmentation}},
%  booktitle = {IEEE International Conference on Computer Vision and Pattern Recognition},
%  year = {2010},
%  month = {June}, 
%  pdf = {http://sminchisescu.ins.uni-bonn.de/papers/cs-cvpr10.pdf 1}
% }
%
% @misc{cpmc-release1,
%  author = "J. Carreira and C. Sminchisescu",
%  title = "Constrained Parametric Min-Cuts for Automatic Object Segmentation, Release 1",
%  howpublished = "http://sminchisescu.uni-bonn.de/code/cpmc-release1/"
% }
%
% (C) Joao Carreira 2010
% 
function [masks, scores] = cpmc_maskonly(exp_dir, directory, img_name, segm_pars, overwrite)
    DefaultVal('segm_pars', []);
    
    if ~isfield(segm_pars,'pb_folder')
		segm_pars.pb_folder = [exp_dir './PB/'];
    end
    if ~isfield(segm_pars,'name')
        segm_pars.name = 'dummy_masks';
    end

        % UniformSegmenter uses a uniform unary term. LongRangeSegmenter
        % uses a color-based unary term. GridOfFramesSegmenter is defined
        % inside subframes (rectangular regions of interest), and it's good for smaller objects.       
        % Each will generate and solve different energy functions.
    if ~isfield(segm_pars,'segm_methods')
        segm_pars.segm_methods = {'UniformSegmenter', 'LongRangeSegmenter', 'GridOfFramesSegmenter'};        
    end
        
        % can set a limit on the number of segments for each kind of
        % Segmenter. The joint set will be filtered in the end.
    if ~isfield(segm_pars,'max_n_segms')
        segm_pars.max_n_segms = [inf inf inf]; 
    end
    if ~isfield(segm_pars,'min_n_pixels')
        segm_pars.min_n_pixels = 200; % 1000
    end
    if ~isfield(segm_pars,'sigma')
        segm_pars.sigma = {1, 2, 0.8};
    end
        %segm_pars.sigma = {1.2, 2.2, 1}; % larger is smoother (fewer segments)
        
        % how much to resize the image (there doesn't seem to be any numerical negative 
        % impact on the VOC segmentation dataset to resize to half).
    if ~isfield(segm_pars,'resize_factor')
        segm_pars.resize_factor= 0.5;
    end
        
        % does a morphological operation to remove little wires in the
        % image. In general a good idea. But might negatively affect wiry structures
        % like bikes a little bit.
    if ~isfield(segm_pars,'morph_open')
        segm_pars.morph_open = true;
    end
        
        % Do the fast filtering step (always recommended).
    if ~isfield(segm_pars,'filter_segments')
        segm_pars.filter_segments = {true, true, true};
    end
        
        % Dimensions of the grid, for each segmenter type.
        %
        % Be careful with the third set. In GridOfFramesSegmenter, a grid
        % is set up inside each subframe . [1 1] sets a single
        % foreground seed, [2 2] would set 4 foreground seeds. If you have 40 subframes like currently, it's
        % really better not to have more than [1 1].
        % On the other two [5 5] should be fine. We used [6 6] on
        % the VOC2010 challenge.
    if ~isfield(segm_pars,'grid_dims')
        segm_pars.grid_dims = {[5 5], [5 5], [1 1]};  
    end
        
        % The maximum energy for the initial filtering step, for each
        % method. The proper value depends on the sigma value, which is
        % unfortunate. If  you don't to spend time on it, just leave the
        % default value. If you don't want to filter based on the energy uncomment the following line        
        %the_segm_pars.max_energy = {inf, inf, inf};        
                
        %%%% these are parameters related to the GridOfFramesSegmenter
        % Two options are implemented: a regular grid of subframes, and
        % using a bounding box detector. Using the detector gets 1% better
        % covering.        
    if ~isfield(segm_pars,'window_gen_parms')
        segm_pars.window_gen_parms.kind = 'grid_sampler';      
        segm_pars.window_gen_parms.det_classes = []; % this needs to be here (bug)
    end
    if ~isfield(segm_pars,'windows_folder')
        segm_pars.windows_folder = [exp_dir 'WindowsOfInterest/grid_sampler'];                
    end
        % If you want to use instead the detector you'll have to install
        % the latent svm code from http://people.cs.uchicago.edu/~pff/latent/, including the star
        % cascade. Then you can uncomment the  two bottom lines
        %segm_pars.window_gen_parms.kind = 'sliding_window_detection';
        %segm_pars.windows_folder = [exp_dir 'WindowsOfInterest/sliding_window_detection']; 
    if ~isfield(segm_pars,'randomize_N')
        segm_pars.randomize_N = 1000; % maximum number of segments to pass to the clustering step, for each type of segmenter. For speed considerations.
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%% 1. compute masks %%%%%%%%%%%%    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    t = tic();
    disp('Starting computation of figure-ground segmentations');
    masks = cpmc_masks(exp_dir, directory, img_name, segm_pars, overwrite);
    time_segm = toc(t);
    fprintf('Time computing figure-ground segmentations: %f\n', time_segm);
end

