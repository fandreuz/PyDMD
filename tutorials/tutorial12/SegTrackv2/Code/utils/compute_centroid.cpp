#include "mex.h"
#include "stdio.h"
#include "time.h"

void compute_centroid(bool *segms, size_t height, size_t width, size_t nsegms, float *centroid_x, float *centroid_y)
{
    int i,j,k;
    long long cent_x, cent_y;
    int idx = 0, area;
    for(k=0;k<nsegms;k++)
    {
        cent_x = cent_y = 0.0;
        area = 0;
        for(i=1;i<=width;i++)
            for(j=1;j<=height;j++)
            {
                if(segms[idx])
                {
                    area++;
                    cent_x += i;
                    cent_y += j;
                }
                idx++;
            }
        centroid_x[k] = cent_x / (float) area / (float) width;
        centroid_y[k] = cent_y / (float) area / (float) height;
    }
}

void mexFunction(int nargout, mxArray *out[], int nargin, const mxArray *in[]) 
{

    /* declare variables */
    mwSize width, height, nsegms;
    mwSize *sizes;
    bool *segms;
    float *o1, *o2;
    
    /* check argument */
    if (nargin<1) {
        mexErrMsgTxt("One input argument required ( the segments in row * column * num_segments form )");
    }
    if (nargout>2) {
        mexErrMsgTxt("Too many output arguments");
    }
    /* sizes */
    sizes = (mwSize *)mxGetDimensions(in[0]);
    height = sizes[0];
    width = sizes[1];
    nsegms = sizes[2];

    if (!mxIsLogical(in[0]) || mxGetNumberOfDimensions(in[0]) != 3) {
        mexErrMsgTxt("Usage: segms must be a 3-dimensional logical matrix");
    }

    segms = (bool *) mxGetData(in[0]);
    out[0] = mxCreateNumericMatrix(nsegms, 1 ,mxSINGLE_CLASS, mxREAL);
    out[1] = mxCreateNumericMatrix(nsegms, 1 ,mxSINGLE_CLASS, mxREAL);
    if (out[0]==NULL || out[1]==NULL) {
	    mexErrMsgTxt("Not enough memory for the output matrix");
    }
    o1 = (float *) mxGetPr(out[0]);
    o2 = (float *) mxGetPr(out[1]);

    compute_centroid(segms, height, width, nsegms, o1, o2);
}
