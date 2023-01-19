/* Copyright (C) 2013 Fuxin Li
 Fast version of segment overlap computation.
 */

/*---
function [o] = segm_overlap_mex(segms)
Input:
    segms - binary 3D matrix, row*column*segments
Output:
    o = symmetric overlap matrix
or

function [o12] = segm_overlap_mex(segms1, segms2)
Input:
	segms1 - binary 3D matrix, row*column*segments_1
	segms2 - binary 3D matrix, row*column*segments_2
Output:
    o12 = segments_1 * segments_2 overlap matrix
--*/

#include "omp.h"
#include "mex.h"
#include "math.h"
#include "time.h"
#include "float.h"

void overlap_w_bb_filter(const bool *segms, const size_t width, const size_t height, const size_t nsegms, float *overlap, const unsigned long *segm_sizes,
         const unsigned short *left, const unsigned short *top, const unsigned short *right, const unsigned short *bottom)
{
    #pragma omp parallel for schedule(dynamic,2)
	for(int i=0;i<nsegms;i++)
	{
		for(int j=i+1;j<nsegms;j++)
		{
		    int index_ij, index_ik, index_jk, res_int;
			unsigned short int_bbleft, int_bbtop, int_bbright, int_bbbottom;
			index_ij = i * nsegms + j;
			res_int = 0;
// No overlap on bounding box means instant solution
			if (right[i] < left[j] || right[j] < left[i] || top[i] > bottom[j] || top[j] > bottom[i])
			{
				overlap[index_ij] = 0.0;
				continue;
			}
// Now find the overlapping part of the bounding box and only compute res_int inside
// For left and top, we take the maximum, for right and bottom, we take the minimum
			int_bbleft = left[i] > left[j] ? left[i] : left[j];
			int_bbtop = top[i] > top[j] ? top[i] : top[j];
			int_bbright = right[i] < right[j] ? right[i] : right[j];
			int_bbbottom = bottom[i] < bottom[j] ? bottom[i] : bottom[j];
			for (int ii= int_bbleft;ii<=int_bbright;ii++)
			{
				index_ik = i * width * height + ii * height + int_bbtop;
				index_jk = j * width * height + ii * height + int_bbtop;
                int this_res = 0;
                #pragma omp parallel for reduction(+:this_res)
				for (int jj = 0;jj<=int_bbbottom - int_bbtop;jj++)
				{
				// Only compute at the beginning
					if (segms[index_ik+jj] & segms[index_jk+jj])
						this_res++;
				}
                res_int += this_res;
			}
// Intersection / (A + B - A \int B)
			overlap[index_ij] = res_int / (float(segm_sizes[i] + segm_sizes[j] - res_int) + FLT_MIN);
			int index_ji = j * nsegms + i;
			overlap[index_ji] = overlap[index_ij];

		}
	}
	for (int i=0;i<nsegms;i++)
		overlap[i * nsegms + i] = 1.0;
}

void overlap_binary(const bool *segms, const bool *segms2, const size_t width, const size_t height, const size_t nsegms, const size_t nsegms2, 
	                float *overlap, const unsigned long *segm_sizes, const unsigned short *left, const unsigned short *top, 
					const unsigned short *right, const unsigned short *bottom, const unsigned long *segm_sizes2, const unsigned short *left2, 
					const unsigned short *top2, const unsigned short *right2, const unsigned short *bottom2)
{
#pragma omp parallel for schedule(dynamic,2)
	for(int i=0;i<nsegms;i++)
	{
		for(int j=0;j<nsegms2;j++)
		{
		    int index_ij, index_ik, index_jk, res_int;
			unsigned short int_bbleft, int_bbtop, int_bbright, int_bbbottom;
			index_ij = j * nsegms + i;
			res_int = 0;
// No overlap on bounding box means instant solution
			if (right[i] < left2[j] || right2[j] < left[i] || top[i] > bottom2[j] || top2[j] > bottom[i])
			{
				overlap[index_ij] = 0.0;
				continue;
			}
// Now find the overlapping part of the bounding box and only compute res_int inside
// For left and top, we take the maximum, for right and bottom, we take the minimum
			int_bbleft = left[i] > left2[j] ? left[i] : left2[j];
			int_bbtop = top[i] > top2[j] ? top[i] : top2[j];
			int_bbright = right[i] < right2[j] ? right[i] : right2[j];
			int_bbbottom = bottom[i] < bottom2[j] ? bottom[i] : bottom2[j];
			for (int ii= int_bbleft;ii<=int_bbright;ii++)
			{
				index_ik = i * width * height + ii * height + int_bbtop;
				index_jk = j * width * height + ii * height + int_bbtop;
                int this_res = 0;
                #pragma omp parallel for reduction(+:this_res)
				for (int jj = 0;jj<=int_bbbottom - int_bbtop;jj++)
				{
				// Only compute at the beginning
					if (segms[index_ik+jj] & segms2[index_jk+jj])
						this_res++;
				}
                res_int += this_res;
			}
// Intersection / (A + B - A \int B)
			overlap[index_ij] = res_int / (float(segm_sizes[i] + segm_sizes2[j] - res_int) + FLT_MIN);
		}
	}
}

/* Compute the bounding boxes for each segment */
/* Not fastest, test if fast enough */
void compute_bounding_box(const bool *segms, size_t width, size_t height, size_t nsegms,
        unsigned long *segm_size, unsigned short *left, unsigned short *top, unsigned short *right, unsigned short *bottom)
{
	int k;
    #pragma omp parallel for schedule(dynamic,2)
	for (k=0;k<nsegms;k++)
	{
		long cur_size = 0;
		short cur_left = width;
		short cur_top = height;
		short cur_right = 0;
        short cur_bottom = 0;
        size_t idx = width * height * k;
		for (int i=0;i<width;i++)
			for (int j=0;j<height;j++)
			{
				if(segms[idx])
				{
					cur_size++;
					if (i < cur_left)
						cur_left = i;
					if (j < cur_top)
						cur_top = j;
					if (i > cur_right)
						cur_right = i;
					if (j > cur_bottom)
						cur_bottom = j;
				}
				idx++;
			}
		segm_size[k] = cur_size;
		if (cur_size == 0)
			left[k] = right[k] = top[k] = bottom[k] = 0;
		else
			left[k] = cur_left, right[k] = cur_right, top[k] = cur_top, bottom[k] = cur_bottom;
	}
}

void mexFunction(int nargout, mxArray *out[], int nargin, const mxArray *in[]) 
{

    /* declare variables */
    mwSize width, height, nsegms, nsegms2;
    mwSize *sizes, *sizes2;
    bool *segms, *segms2;
    float *o;
    
    /* check argument */
    if (nargin<1) {
        mexErrMsgTxt("One input argument required ( the segments in row * column * num_segments form )");
    }
    if (nargout>1) {
        mexErrMsgTxt("Too many output arguments");
    }
    if (!mxIsLogical(in[0]) || mxGetNumberOfDimensions(in[0]) < 2) {
        mexErrMsgTxt("Usage: segms must be a 3-dimensional logical matrix (if 2D, then it's considered h*w*1)");
    }
    /* sizes */
    sizes = (mwSize *)mxGetDimensions(in[0]);
    height = sizes[0];
    width = sizes[1];
	if (mxGetNumberOfDimensions(in[0]) == 3)
		nsegms = sizes[2];
	else
		nsegms = 1;

	if (nargin > 1)
	{
		if (!mxIsLogical(in[1]) || mxGetNumberOfDimensions(in[1]) < 2) {
			mexErrMsgTxt("Usage: segms2 must be a 3-dimensional logical matrix (if 2D, then it's considered h*w*1)");
		}
		sizes2 = (mwSize *)mxGetDimensions(in[1]);
		if (sizes2[0] != height || sizes2[1] != width)
			mexErrMsgTxt("The two segment matrices must be from the same image (height and width must be the same)!");
		if (mxGetNumberOfDimensions(in[1]) == 3)
			nsegms2 = sizes2[2];
		else
			nsegms2 = 1;
		segms2 = (bool *) mxGetData(in[1]);
	}

    segms = (bool *) mxGetData(in[0]);
	if (nargin > 1)
		out[0] = mxCreateNumericMatrix(nsegms, nsegms2,mxSINGLE_CLASS, mxREAL);
	else
		out[0] = mxCreateNumericMatrix(nsegms, nsegms,mxSINGLE_CLASS, mxREAL);

    if (out[0]==NULL) {
	    mexErrMsgTxt("Not enough memory for the output matrix");
    }
    o = (float *) mxGetPr(out[0]);

	/* Not expecting an image has larger dimensions than 65536 * 65536 */
	unsigned short *bbleft = new unsigned short [nsegms];
	unsigned short *bbtop = new unsigned short [nsegms];
	unsigned short *bbright = new unsigned short [nsegms];
	unsigned short *bbbottom = new unsigned short [nsegms];
	unsigned long *segm_sizes = new unsigned long [nsegms];
	compute_bounding_box(segms, width, height, nsegms, segm_sizes, bbleft, bbtop, bbright, bbbottom);
	if (nargin > 1)
	{
		unsigned short *bbleft2 = new unsigned short [nsegms2];
		unsigned short *bbtop2 = new unsigned short [nsegms2];
		unsigned short *bbright2 = new unsigned short [nsegms2];
		unsigned short *bbbottom2 = new unsigned short [nsegms2];
		unsigned long *segm_sizes2 = new unsigned long [nsegms2];
		compute_bounding_box(segms2, width, height, nsegms2, segm_sizes2, bbleft2, bbtop2, bbright2, bbbottom2);
		overlap_binary(segms, segms2, width, height, nsegms, nsegms2, o, segm_sizes, bbleft, bbtop, bbright, bbbottom, segm_sizes2, bbleft2, bbtop2, bbright2, bbbottom2);
		delete[] bbleft2;
		delete[] bbright2;
		delete[] bbtop2;
		delete[] bbbottom2;
		delete[] segm_sizes2;
	}
	else
		overlap_w_bb_filter(segms, width, height, nsegms, o, segm_sizes, bbleft, bbtop, bbright, bbbottom);
        
	delete[] bbleft;
	delete[] bbright;
	delete[] bbtop;
	delete[] bbbottom;
	delete[] segm_sizes;
}
