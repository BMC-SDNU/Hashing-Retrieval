#include "mex.h"
#include <float.h>
#include <memory.h>
#include <stdio.h>
#include <emmintrin.h> // for float
#include <omp.h>
// #define __SSE2__

static inline double chi2_baseline_double(const int n, const double* const x, const double* const y) {
    double result = 0.f;
    int i;
    for (i=0; i<n; i++) {
        const double num = x[i]-y[i];
        const double denom = 1./(x[i]+y[i]+DBL_MIN);
        result += num*num*denom;
    }
    return result;
}


/* use compiler intrinsics for 2x parallel processing */
static inline double chi2_intrinsic_double(int n, const double* x, const double* y) {
    double result=0;
    const __m128d eps = _mm_set1_pd(DBL_MIN);
    const __m128d zero = _mm_setzero_pd();
    __m128d chi2 = _mm_setzero_pd();    

    for ( ; n>1; n-=2) {
        const __m128d a = _mm_loadu_pd(x);
        const __m128d b = _mm_loadu_pd(y);
	x+=2;
	y+=2;
        const __m128d a_plus_b = _mm_add_pd(a,b);
        const __m128d a_plus_b_plus_eps = _mm_add_pd(a_plus_b,eps);
        const __m128d a_minus_b = _mm_sub_pd(a,b);
        const __m128d a_minus_b_sq = _mm_mul_pd(a_minus_b, a_minus_b);
        const __m128d quotient = _mm_div_pd(a_minus_b_sq, a_plus_b_plus_eps);
        chi2 = _mm_add_pd(chi2, quotient);
    }
    const __m128d shuffle = _mm_shuffle_pd(chi2, chi2, _MM_SHUFFLE2(0,1));
    const __m128d sum = _mm_add_pd(chi2, shuffle);
// with SSE3, we could use hadd_pd, but the difference is negligible 

    _mm_store_sd(&result,sum);
    _mm_empty();
    if (n)
        result += chi2_baseline_double(n, x, y); // remaining entries
    return result;
}


/* calculate the chi2-distance between two vectors/histograms */
double chi2_double(const int dim, const double* const x, const double* const y) {
    double (*chi2_double)(const int, const double*, const double*) = chi2_baseline_double;
#ifdef __SSE2__
    chi2_double = chi2_intrinsic_double;
#endif
    return chi2_double(dim, x, y);
}

/* calculate the chi2-measure between two sets of vectors/histograms */
double chi2sym_distance_double(const int dim, const int nx, const double* const x, 
                               double* const K) {
    double (*chi2_double)(const int, const double*, const double*) = chi2_baseline_double;
#ifdef __SSE2__
    chi2_double = chi2_intrinsic_double;
#endif

    double sumK=0.;
#pragma omp parallel 
    {
        int i,j;
#pragma omp for reduction (+:sumK) schedule (dynamic, 2)
        for (i=0;i<nx;i++) {
    	    K[i*nx+i]=0.;
            for (j=0;j<i;j++) {
                const double chi2 = chi2_double(dim, &x[i*dim], &x[j*dim]);
    	    	K[i*nx+j] = chi2;
	    	    K[j*nx+i] = chi2;    
        		sumK += 2*chi2;
            }
	    }
    }
    return sumK/((float)(nx*nx)); 
}

/* calculate the chi2-measure between two sets of vectors/histograms */
double chi2_distance_double(const int dim, const int nx, const double* const x, 
                                         const int ny, const double* const y, double* const K) {
    double (*chi2_double)(const int, const double*, const double*) = chi2_baseline_double;
#ifdef __SSE2__
    chi2_double = chi2_intrinsic_double;
#endif

    double sumK=0.;
#pragma omp parallel 
    {
        int i,j;
#pragma omp for reduction (+:sumK)
        for (i=0;i<nx;i++)
            for (j=0;j<ny;j++) {
                const double chi2 = chi2_double(dim, &x[i*dim], &y[j*dim]);
		K[i*ny+j] = chi2;
		sumK += chi2;
	    }
    }
    return sumK/((float)(nx*ny)); 
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	double	*px, *pxn, *pl;
    const double	*X, *L;
	double			*pK;
	int				npts, dim, nlandmarks;	
	int				i, j, k, l;

	if (nrhs != 2)
		mexErrMsgTxt("two input arguments expected.");
	if (nlhs != 1)
		mexErrMsgTxt("one output arguments expected.");

	if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) ||
		mxGetNumberOfDimensions(prhs[0]) != 2)
		mexErrMsgTxt("input 1 (X) must be a real double matrix");

	dim = mxGetM(prhs[0]);
	npts = mxGetN(prhs[0]);

	if (!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) ||
		mxGetNumberOfDimensions(prhs[1]) != 2 ||
		mxGetM(prhs[1]) != dim)
		mexErrMsgTxt("input 2 (L) must be a real double matrix compatible with input 1 (X)");
    nlandmarks =  mxGetN(prhs[1]);   
  
	plhs[0] = mxCreateDoubleMatrix(nlandmarks, npts, mxREAL);
	pK = mxGetPr(plhs[0]);      
    
	X = mxGetPr(prhs[0]);
    L = mxGetPr(prhs[1]);
  
    chi2_distance_double(dim, npts, X, nlandmarks, L, pK);
}
