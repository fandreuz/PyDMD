#!/bin/bash
MATLAB_DIR = /opt/matlab/2013a

all:
	gcc assign_superpix.cpp -o assign_superpix.mexa64 -shared -fPIC -I$(MATLAB_DIR)/extern/include -L$(MATLAB_DIR)/bin/glnxa64 -Wl,-rpath-link,$(MATLAB_DIR)/bin/glnxa64 -lm -lmx -lmex -lmat -lmwservices -lut -std=c++0x

clean:
	rm assign_superpix.mexa64

