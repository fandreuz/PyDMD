This is a optimized version for CPMC in windows platform. The main changes are
1. replace GPb with (modified) efficient boundary detector
2. newer version of VLfeat and ColorSIFT
3. revised CPP files that are windows friendly (removing all posix functions) and a new file that compiles everything
4. clean up of the code



To run 
1. run myCompile.m for all mex files (64bit win binaries are already included)
1.1 set your folder of Boost in myCompile.m
1.2 check your compiler for openmp support
2. run cpmc_example.m for a demo, takes 3 mins on my laptop for the whole pipeline (segmentation + ranking)
3. results are slightly different of the original version (mainly at boundary due to gPb)

Also see CPMC release notes in /cpmc_singleframe.

Yin Li from Georgia Tech, 2013.