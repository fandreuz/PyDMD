#include "mex.h"
#include <algorithm>
#include <functional>
#include <cmath>

void get_top_k_avg(double *x, int xylen, double *ret, const int k)
{
    int i;
    double *pred_vals = new double[k];
    double tmp = 0.0;
// First 1-k is different, take plain average
    for(i=0;i<k;i++)
    {
        if(i==xylen)
            break;
        tmp += x[i];
        ret[i] = tmp / k;
        pred_vals[i] = x[i];
    }
    if (i==xylen)
    {
        delete[] pred_vals;
        return;
    }
// Now, tmp is the total of top k
// Make a heap out of it
// x_pred is sorted in descending order therefore go with it
    std::make_heap(&pred_vals[0],&pred_vals[k], std::greater<double>());
    for(i=k;i<xylen;i++)
    {
// In this case, join it in the top-k heap, and change the avg value
        if (x[i] > pred_vals[0])
        {
            std::pop_heap(&pred_vals[0], &pred_vals[k], std::greater<double>());
            tmp += x[i] - pred_vals[k-1];
            pred_vals[k-1] = x[i];
            std::push_heap(&pred_vals[0], &pred_vals[k], std::greater<double>());
            ret[i] = tmp / k;
        }
        else
            ret[i] = ret[i-1];
    }
    delete[] pred_vals;
}

// 2 rhs: True Quality (sorted in descending order of predicted quality), length_to_do
// Asking for MATLAB sort because it is slightly better than C/C++ sort because it returns the indices, thus
// making it easier to handle two arrays.
// lhs: Top-k quality
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    double *x;
    int k, len;
    if (nrhs < 2)
        k = 5;
    else
        k = (unsigned int)(mxGetPr(prhs[1])[0]);
    len = mxGetN(prhs[0]) > mxGetM(prhs[0])? mxGetN(prhs[0]):mxGetM(prhs[0]);
    double *ret = (double *) mxMalloc(sizeof(double) * len);
    if (mxGetClassID(prhs[0]) != mxDOUBLE_CLASS)
    {
        mexPrintf("Single input/outputs not yet supported for now, will add support later.");
        return;
    }
    x = mxGetPr(prhs[0]);
    get_top_k_avg(x, len, ret, k);
    plhs[0] = mxCreateDoubleMatrix(len,1,mxREAL);
    mxSetPr(plhs[0], ret);
}