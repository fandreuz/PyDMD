// let us forget about the fancy SSE2
// it is tricky to wrtie / read
// it is somehow hard to compile in win using mex

#include <stdio.h>
#include <mex.h>
#include <cfloat>

/* OpenMP allows to achieve almost linear speedup on multiCore CPUs: use gcc-4.2 -fopenmp */
//#ifdef _OPENMP
#include <omp.h>
//#endif

float chi2_baseline_float(const int n, const float* x, const float* y) {
    float result = 0.f;
    int i;
    for (i=0; i<n; i++) {
        const float num = x[i]-y[i];
        const float denom = 1.0f/(x[i]+y[i]+FLT_MIN);
        result += num*num*denom;
    }
    return result;
}


/* calculate the chi2-distance matrix between a sets of vectors/histograms. */
float chi2sym_distance_float(const int dim, const int nx, const float* const x, float* const K) 
{
    float sumK=0.f;
#pragma omp parallel
    {
        int i,j;
#pragma omp for reduction (+:sumK) schedule (dynamic,2)
        for (i=0;i<nx;i++) {
    	    K[i*nx+i]=0.;
            for (j=0;j<i;j++) {
	    	const float chi2 = chi2_baseline_float(dim, &x[i*dim], &x[j*dim]); 
                K[i*nx+j] = chi2;
                K[j*nx+i] = chi2;
	        	sumK += 2*chi2;
            }
	    }
    }
    return sumK/((float)(nx*nx)); 
}

/* calculate the chi2-distance matrix between two sets of vectors/histograms. */
float chi2_distance_float(const int dim, const int nx, const float* const x, const int ny, const float* const y, float* const K) 
{
    float sumK=0.f;
#pragma omp parallel
    {
        int i,j;
#pragma omp for reduction (+:sumK) schedule (dynamic,2)
        for (i=0;i<nx;i++) {
            for (j=0;j<ny;j++) {
	    	float chi2 = chi2_baseline_float(dim, &x[i*dim], &y[j*dim]); 
                K[i*ny+j] = chi2;
        		sumK += chi2;
            }
	    }
    }
    return sumK/((float)(nx*ny)); 
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	float *vecA, *vecB, *dist;
	int dim, ptsA, ptsB, *sym;
	mxClassID the_class;
	mwSize ndims[2];
	mxArray* mxdist;

	if (nrhs == 0)
	{
		mexPrintf("Usage: d = chi2_mex(X,Y);\n");
		mexPrintf("where X and Y are matrices of dimension [dim,npts]\n");
		mexPrintf("\nExample\n a = rand(2,10);\n b = rand(2,20);\n d = chi2_mex(a,b);\n");
		return;
	}

	if (nrhs != 3){
		mexPrintf("three input arguments expected: A, B, and sym, sym says wether A and B are the same");
		return;
	}

	if (mxGetNumberOfDimensions(prhs[0]) != 2 || mxGetNumberOfDimensions(prhs[1]) != 2)
	{
		mexPrintf("inputs must be two dimensional");
		return;
	}
    
    
    the_class = mxGetClassID(prhs[0]);
    
    if(the_class != mxSINGLE_CLASS) {
        mexErrMsgTxt("Histograms should have single precision!\n");
    }
            

	vecA = (float *)mxGetPr(prhs[0]);
	vecB = (float *)mxGetPr(prhs[1]);
    sym = (int *) mxGetPr(prhs[2]);
    
	ptsA = mxGetN(prhs[0]);
	ptsB = mxGetN(prhs[1]);
	dim = mxGetM(prhs[0]);

	if (dim != mxGetM(prhs[1]))
	{
		mexPrintf("Dimension mismatch");
		return;
	}

    ndims[0] = ptsA;
	ndims[1] = ptsB;
    
    mxdist = mxCreateNumericArray(2, ndims,the_class,mxREAL);    
    dist = (float *)mxGetData(mxdist);
	/*plhs[0] = mxCreateDoubleMatrix(ptsA,ptsB,mxREAL);*/
	/*dist = (float *)mxGetPr(plhs[0]);*/
    
	/*chi2_distance_float(dim,ptsB,vecB,ptsA,vecA,dist);    */
          
/* printf("get_num_threads: %d\n", omp_get_num_threads()); 
 printf("hello");*/
 
    if(*sym) {
        chi2sym_distance_float(dim,ptsB,vecB, dist); 
    } else {
        chi2_distance_float(dim,ptsB,vecB,ptsA,vecA,dist); 
    }
            
    plhs[0] = mxdist;
    
	return;
}
