// boot.cpp
// c++ source code for creating boot.mex file using mex as follows:
// 
// mex boot.cpp
//
// boot.mex is a function file for generating balanced bootstrap sample indices
// or for generating balanced bootstrap resamples of a data vector
//
// USAGE
// BOOTSAM = boot (N, NBOOT)
// BOOTSAM = boot (X, NBOOT)
// BOOTSAM = boot (..., NBOOT, LOO)
// BOOTSAM = boot (..., NBOOT, LOO, SEED)
// BOOTSAM = boot (..., NBOOT, LOO, SEED, WEIGHTS)
//
// INPUT VARIABLES
// N (double) is the number of rows (of the data vector)
// X (double) is a data vector intended for resampling
// NBOOT (double) is the number of bootstrap resamples
// LOO (boolean) to set the resampling method: false (for bootstrap) or true
//   (for bootknife)
// SEED (double) is a seed used to initialise the pseudo-random number generator
// WEIGHTS (double) is a weight vector of length N
//
// OUTPUT VARIABLE
// BOOTSAM (double) is an N x NBOOT matrix of sample indices (N) or NBOOT 
//   columns of resampled data (X)
//
// NOTES
// LOO is an optional input argument. The default is false. If LOO is true
// bootknife resampling is used, which involves creating leave-one-out
// jackknife samples of size N - 1, and then drawing resamples of size N with
// replacement from the jackknife samples, thereby incorporating Bessel's
// correction into the resampling procedure. The sample index for omission in
// each bootknife resample is selected systematically. When the remaining number
// of bootknife resamples is not divisible by the sample size (N), then the
// sample index omitted is selected randomly. Balanced resampling only applies
// when NBOOT > 1.
// SEED is an optional scalar input argument used to initialize the random
// number generator to make resampling reproducible between calls to boot.
// WEIGHTS is an optional input argument. If WEIGHTS is empty or not provided,
// the default is a vector of each element equal to NBOOT (i.e. uniform
// weighting). Each element of WEIGHTS is the number of times that the
// corresponding index is represented in bootsam. Therefore, the sum of WEIGHTS
// should equal N * NBOOT. 
//
// Compared to previous versions (in package versions <=5.6.0), the boot.mex 
// function is now thread safe.
//
// Requirements: Compilation requires C++11
//
// Author: Andrew Charles Penn (2022)

#include "mex.h"
#include <vector>
#include <random>
using namespace std;


void mexFunction (int nlhs, mxArray* plhs[],
                  int nrhs, const mxArray* prhs[]) 
{

    // Input variables
    if ( nrhs < 2 ) {
        mexErrMsgTxt ("At least two input arguments are required.");
    }
    if ( nrhs > 5) {
        mexErrMsgTxt ("Too many input arguments.");
    } 
    // First input argument (n or x)
    double *x = (double *) mxGetData (prhs[0]);
    long unsigned int n = mxGetNumberOfElements (prhs[0]);
    bool isvec;
    if ( n > 1 ) {
        const mwSize *sz = mxGetDimensions (prhs[0]);
        if ( sz[0] > 1 && sz[1] > 1 ) {
            mexErrMsgTxt ("The first input argument must be either a scalar (N) or vector (X).");
        }
        isvec = true;
    } else {
        isvec = false;
        if ( mxIsComplex (prhs[0]) ) {
            mexErrMsgTxt ("The first input argument (N) cannot contain an imaginary part.");
        }
        if ( *x != static_cast<long unsigned int>(*x) ) {
            mexErrMsgTxt ("The value of the first input argument (N) must be a positive integer.");
        }
        if ( !mxIsFinite (*x) ) {
            mexErrMsgTxt ("The first input argument (N) cannot be NaN or Inf.");
        }
        n = static_cast<long unsigned int>(*x);
    }
    if ( !mxIsClass (prhs[0], "double") ) {
        mexErrMsgTxt ("The first input argument (N or X) must be of type double.");
    }
    // Second input argument (nboot)
    const long unsigned int nboot = static_cast<const long unsigned int> ( *(mxGetPr (prhs[1])) );
    if ( mxGetNumberOfElements (prhs[1]) > 1 ) {
        mexErrMsgTxt ("The second input argument (NBOOT) must be scalar.");
    }
    if ( !mxIsClass (prhs[1], "double") ) {
        mexErrMsgTxt ("The second input argument (NBOOT) must be of type double.");
    }
    if ( mxIsComplex (prhs[1]) ) {
        mexErrMsgTxt ("The second input argument (NBOOT) cannot contain an imaginary part.");
    }
    if ( nboot <= 0 ) {
        mexErrMsgTxt ("The second input argument (NBOOT) must be a positive integer.");
    }
    if ( !mxIsFinite (nboot) ) {
        mexErrMsgTxt ("The second input argument (NBOOT) cannot be NaN or Inf.");    
    }
    // Third input argument (LOO)
    bool loo;
    if ( nrhs > 2 && !mxIsEmpty (prhs[2]) ) {
        if (mxGetNumberOfElements (prhs[2]) > 1 || !mxIsClass (prhs[2], "logical")) {
            mexErrMsgTxt ("The third input argument (LOO) must be a logical scalar value.");
        }
        loo = static_cast<bool> ( *(mxGetLogicals (prhs[2])) );
    } else {
        loo = false;
    }
    // Fourth input argument (seed)
    unsigned int seed;
    if ( nrhs > 3 && !mxIsEmpty (prhs[3]) ) {
        if ( mxGetNumberOfElements (prhs[3]) > 1 ) {
            mexErrMsgTxt ("The fourth input argument (SEED) must be a scalar value.");
        }
        if ( !mxIsClass (prhs[3], "double") ) {
            mexErrMsgTxt ("The fourth input argument (SEED) must be of type double.");
        }
        seed = static_cast<unsigned int> ( *(mxGetPr(prhs[3])) );
        if ( !mxIsFinite (seed) ) {
            mexErrMsgTxt ("The fourth input argument (SEED) cannot be NaN or Inf.");    
        }
    } else {
        random_device rd;
        seed = static_cast<unsigned int> ( rd () );
    }
    // Fifth input argument (w, weights)
    // Error checking is handled later (see below in 'Declare variables' section) 

    // Output variables
    if (nlhs > 1) {
        mexErrMsgTxt ("Too many output arguments.");
    }

    // Declare variables
    mwSize dims[2] = {static_cast<mwSize>(n), static_cast<mwSize>(nboot)};
    plhs[0] = mxCreateNumericArray (2, dims, 
                mxDOUBLE_CLASS, 
                mxREAL);                   // Prepare array for sample indices
    long long unsigned int N = n * nboot;  // Total counts of all sample indices
    long long unsigned int k;              // Variable to store random number
    long long int d;                       // Counter for cumulative sum calculation
    vector<long long int> c(n, nboot);     // Counter for each of the sample indices
    if ( nrhs > 4 && !mxIsEmpty (prhs[4]) ) {
        // Assign user defined weights (counts)
        if ( !mxIsClass (prhs[4], "double") ) {
            mexErrMsgTxt ("The fifth input argument (WEIGHTS) must be of type double.");
        }
        if ( mxIsComplex (prhs[4]) ) {
            mexErrMsgTxt ("The fifth input argument (WEIGHTS) cannot contain an imaginary part.");
        }
        double *w = (double *) mxGetData (prhs[4]);
        if ( mxGetNumberOfElements (prhs[4]) != n ) {
            mexErrMsgTxt ("WEIGHTS must be a vector of length N or be the same length as X.");
        }
        long long int s = 0; 
        for ( int i = 0; i < n ; i++ )  {
            if ( !mxIsFinite (w[i]) ) {
                mexErrMsgTxt ("The fifth input argument (WEIGHTS) cannot contain NaN or Inf.");    
            }
            if ( w[i] < 0 ) {
                mexErrMsgTxt ("The fifth input argument (WEIGHTS) must contain only positive integers.");
            }
            c[i] = w[i]; // Set each element in c to the specified weight    
            s += c[i];
        }
        if ( s != N ) {
            mexErrMsgTxt ("The elements of WEIGHTS must sum to N * NBOOT.");
        }
    }
    long long int m = 0;  // Counter for LOO sample index r
    int r = -1;           // Sample index for LOO

    // Create pointer so that we can access elements of bootsam (i.e. plhs[0])
    double *ptr = (double *) mxGetData(plhs[0]);

    // Initialize pseudo-random number generator (Mersenne Twister 19937)
    mt19937_64 rng (seed);
    uniform_int_distribution<long long unsigned int> distr (0, n - 1);
    uniform_int_distribution<long long unsigned int> distk (0, N - 1);

    // Perform balanced sampling
    for ( int b = 0; b < nboot ; b++ ) { 
        if ( loo == true ) {
            // Note that the following division operations are for integers 
            if ( (b / n) == (nboot / n) ) {
                r = distr (rng);      // random
            } else {
                r = b - (b / n) * n;  // systematic
            }
            m = c[r];
            c[r] = 0;
        }
        for ( int i = 0; i < n ; i++ ) {
            if ( loo == true ) {
                // Only leave-one-out if sample index r doesn't account for all
                // remaining sampling counts
                if (N == m) {
                    c[r] = m;
                    m = 0;
                    loo = false;
                }
            }
            distk.param (uniform_int_distribution<long long unsigned int>::param_type (0, N - m - 1));
            k = distk (rng); 
            d = c[0];
            for ( int j = 0; j < n ; j++ ) { 
                if ( k < d ) {
                    if (isvec) {
                        ptr[b * n + i] = x[j];
                    } else {
                        ptr[b * n + i] = j + 1;
                    }
                    if ( nboot > 1 ) {
                      c[j] -= 1;
                      N -= 1;
                    }
                    break;
                } else {
                    d += c[j + 1];
                }
            }
        }
        if ( loo == true ) {
            c[r] = m;
            m = 0;
        }
    }

    return;

}
