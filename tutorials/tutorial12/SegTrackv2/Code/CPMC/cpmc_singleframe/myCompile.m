% segm with openmp support (cpp file modified)
mex -O -v code/segm_intersection_mex.c COMPFLAGS="$COMPFLAGS /openmp" -output code/segm_intersection_mex
mex -O -v code/segm_overlap_mex.c COMPFLAGS="$COMPFLAGS /openmp " -output code/segm_overlap_mex

% other mex file
mex -O -v code/cartprod_mex.c -output code/cartprod_mex
mex -O -v code/int_hist.c -output code/int_hist
mex -O -v code/intens_pixel_diff_mex.c -output code/intens_pixel_diff_mex
mex -O -v external_code/overlap_care.c -output external_code/overlap_care
mex -O -v external_code/imrender/vgg/vgg_segment_gb.cxx -output external_code/imrender/vgg/vgg_segment_gb

% for paraFmex (cpp file modified)
mex  -O -v -DBREAKPOINTS external_code/paraFmex/hoch_pseudo_par.c -output external_code/paraFmex/hoch_pseudo_par
mex -O -v external_code/paraFmex/paraFmex.cpp external_code/paraFmex/joao_parser.cpp external_code/paraFmex/Graph.cpp external_code/paraFmex/Solver.cpp external_code/paraFmex/stdafx.cpp -output external_code/paraFmex/paraFmex

% for ranking (cpp file modified)
mex -O -v external_code/my_phog_desc_mex.cpp -output external_code/my_phog_desc_mex -IE:\boost_1_51\boost_1_51_0               % change the folder to your boost lib
mex -O -v external_code/mpi-chi2-v1_5/chi2_mex.c COMPFLAGS="/openmp $COMPFLAGS" -output external_code/mpi-chi2-v1_5/chi2_mex
