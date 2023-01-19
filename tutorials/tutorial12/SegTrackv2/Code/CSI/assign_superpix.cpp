#include <unordered_set>
#include <vector>
#include <string>
#include <cmath>
#include "mex.h"

using namespace std;

// Input is a matrix with width*height rows and nsegms columns, so that we can take each column as a bitset and hash it.
// 2 outputs, masks, the bitset specifying a superpixel, and map, a mapping from each superpixel to the bitset
// This is not fastest yet, since it's doing a copy construction to bitset, best should be doing this in-place
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    bool *x;
    int nsegms, wh;
    double *idx;
	bool *map;
	int i,j,k;
	vector<string> mvd_strings;
    if (nrhs < 1 || mxGetClassID(prhs[0]) != mxLOGICAL_CLASS)
    {
        mexPrintf("assign_superpix takes only 1 boolean matrix!");
        return;
    }
    x = (bool *) mxGetData(prhs[0]);
    wh = mxGetN(prhs[0]);
    nsegms = mxGetM(prhs[0]);
    unordered_set<string> hash_map(1000);
	mvd_strings.reserve(wh);

    if (nlhs > 1)
    {
        plhs[1] = mxCreateDoubleMatrix(wh,1,mxREAL);
        idx = mxGetPr(plhs[1]);
    }
	// Create bitset and insert into hash map, the indices are already determined by the insertion result?
    for( i=0;i<wh;i++)
    {
		string this_pix;
		// Pre-allocate memory
		this_pix.reserve((int)ceil(nsegms/8.0));
		int my_idx = 0;
		for ( j=0;j<((int)ceil(nsegms/8.0));j++)
		{
			char c = 0;
			for(k=0;k<8;k++)
			{
				if(x[i*nsegms+my_idx])
					c += (x[i*nsegms+my_idx] << k);
				my_idx++;
				if (my_idx >= nsegms)
					break;
			}
			this_pix += c;
			if(k<8)
				break;
		}
//        vector<bool> bs(&x[i*nsegms], &x[(i+1)*nsegms]);
        hash_map.insert(this_pix);
		mvd_strings.push_back(this_pix);
    }
	// After insertion, we know how many elements do we have
	plhs[0] = mxCreateLogicalMatrix(hash_map.size(),nsegms);
	map = (bool *) mxGetData(plhs[0]);
	int *bucks = new int[hash_map.bucket_count()];
	int counter = 0;
	// Find the size for each bucket, as well as putting the results in the first output
    for ( i=0;i<hash_map.bucket_count();i++)
    {
		bucks[i] = counter;
        for (unordered_set<string>::local_iterator local_it = hash_map.begin(i);local_it != hash_map.end(i);local_it++)
		{
			// That's a bitset
			int counter2 = 0;
			// Copy from vector of strings
			for (string::const_iterator local_it2 = (*local_it).begin();local_it2 != (*local_it).end();local_it2++)
			{
				if (*local_it2)
				{
					for(int k=0;k<8;k++)
						map[counter*nsegms+counter2 + k] = (*local_it2 >> k) & 1;
				}
				counter2+=8;
			}
			counter++;
		}
    }
	if (nlhs > 1)
	{
	// Now search each bs in the hash map
		for ( i=0;i<wh;i++)
		{
			int bkt = hash_map.bucket(mvd_strings[i]);
			unordered_set<string>::const_iterator the_it = hash_map.find(mvd_strings[i]);
			int counter2 = 0;
		// Find the location within the bin
			for (unordered_set<string>::const_local_iterator local_it = hash_map.begin(bkt);&(*local_it) !=&(*the_it);local_it++)
				counter2++;
			idx[i] = bucks[bkt] + counter2 + 1;
		}
	}
	delete[] bucks;
}