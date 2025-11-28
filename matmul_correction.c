
#include <stdlib.h>
#include <stdio.h>
#include <cblas.h>
#include <tgmath.h>

#include <sys/time.h>

struct timeval timer;

void init_timing(struct timeval* tstart)
{
	gettimeofday(tstart, NULL);
}

float elapsed_time(struct timeval tstart)
{
	struct timeval tmp;
	long long diff;
	gettimeofday(&tmp, NULL);
	diff = tmp.tv_usec - tstart.tv_usec;
	diff += (tmp.tv_sec - tstart.tv_sec) * 1000000;
	return ((float)diff*1.0e-6);
}


void create_matrices(float** A, float** B, float** C, float** C_blas, int M, int N, int K)
{
    size_t i;

    *A = (float*) calloc(M*K,sizeof(float));
    *B = (float*) calloc(K*N,sizeof(float));
    *C = (float*) calloc(M*N,sizeof(float));
    *C_blas = (float*) calloc(M*N,sizeof(float));

    for(i = 0; i < M*K; i++)
        (*A)[i] = ((float)rand()/RAND_MAX-0.5)*0.1;

    for(i = 0; i < K*N; i++)
        (*B)[i] = ((float)rand()/RAND_MAX-0.5)*0.1;
}


void matmul_blas(const float* A, const float* B, float* C, int M, int N, int K)
{
    float alpha = 1.0f, beta = 0.0f;

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
        M, N, K, alpha, A, M, B, K, beta, C, M);
}


//Naive triple loop matmul
void matmul_v1(const float *A, const float *B, float *C, int M, int N, int K)
{
	int i,j,k;
	double acc;
	
	#pragma omp parallel for schedule(dynamic, 1) private(i,j,k)
	for(j = 0; j < N; j++)
		for(i = 0; i < M; i++)	
			for(k = 0; k < K; k++)
				C[j*M+i] += A[k*M+i] * B[j*K+k];
}


//v1 + simple register accumulate
void matmul_v2(const float *A, const float *B, float *C, int M, int N, int K)
{
	int i,j,k;
	float acc;
	
	#pragma omp parallel for schedule(dynamic, 1) private(i,j,k, acc)
	for(j = 0; j < N; j++)
		for(i = 0; i < M; i++)	
		{	
			acc = 0.0;
			for(k = 0; k < K; k++)
				acc += A[k*M+i] * B[j*K+k];
			C[j*M+i] = acc;
		}
}

//v2 + transposed A 
//With the proper set of compilation optimization flags
//this function will be auto-vectorized
void matmul_v3(const float *A, const float *B, float *C, int M, int N, int K)
{
	int i,j,k;
	float *t_A = (float*) malloc(M*K*sizeof(float));
	float acc;
	
	for(i = 0; i < M; i++)	
		for(k = 0; k < K; k++)
			t_A[i*K+k] = A[k*M+i];
	
	#pragma omp parallel for schedule(dynamic, 1) private(i,j,k, acc)
	for(j = 0; j < N; j++)
		for(i = 0; i < M; i++)
		{	
			acc = 0.0;
			for(k = 0; k < K; k++)
				acc += t_A[i*K+k] * B[j*K+k];
			C[j*M+i] = acc;
		}
}

//v3 + manual vectorization through memory aligned SIMD operations 
//Note, require C11 ISO, not compatible with C99 while auto vectorization is compatible

// Define a memory aligned vector type of 32 Bytes -> 8 float of 4 bytes (32 bits) each
typedef float vec __attribute__ (( vector_size(32) ));

void matmul_v4(const float *A, const float *B, float *C, int M, int N, int K)
{
	int i,j,k;
	// Rounded number of 8-element vec in a K
	if(K % 8 != 0)
	{
		printf("Error, mismatch between matrix size and kernel size");
		exit(EXIT_FAILURE);
	}
	
	int n_K = K/8;
	float f_acc;
	
	
	vec *a = (vec*) aligned_alloc(sizeof(vec),M*n_K*sizeof(vec));
	vec *b = (vec*) aligned_alloc(sizeof(vec),N*n_K*sizeof(vec));
	
	for(i = 0; i < M; i++)	
		for(j = 0; j < K; j++)
			a[i*n_K + j/8][j%8] = A[j*M+i];
	
	for(i = 0; i < N; i++)
		for(j = 0; j < K; j++)
			b[i*n_K + j/8][j%8] = B[i*K+j];
	
	#pragma omp parallel for schedule(dynamic, 1) private(i,j, k, f_acc)
	for(j = 0; j < N; j++)
		for(i = 0; i < M; i++)
		{
			vec acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
			for(k = 0; k < n_K; k++)
				acc += a[i*n_K+k] * b[j*n_K+k];
			
			f_acc = 0.0;
			for(k = 0; k < 8; k++)
				f_acc += acc[k];
			C[j*M+i] = f_acc;
		}
}




#define ker_w 6
#define ker_h 16
//ker_h must be a multiple of 8

void kernel(const float *_a, const float *_b, float *_c, 
	int M, int N, int K, int i, int j, int k_start, int k_stop) 
{
	int k, b_m, b_n, l;
	float val;
	//declared in CPU register
	vec t[ker_h/8][ker_w] = {};

	for(k = k_start; k < k_stop; k++)
	{
		for(b_n = 0; b_n < ker_w; b_n++) 
	   	{
			//brodcast B value to the full vector
			val = _b[(i+b_n)*K+k];
			vec beta = {val, val, val, val, val, val, val, val};
			
			vec alpha;
			for(b_m = 0; b_m < ker_h/8; b_m++)
			{
				for(l = 0; l < 8; l++)
					alpha[l] = _a[j+k*M+b_m*8+l];
				t[b_m][b_n] += alpha * beta; // converts to an fma
			}
		}
	}

	// write the results back to C
	for(b_n = 0; b_n < ker_w; b_n++)
		for(b_m = 0; b_m < ker_h/8; b_m++)
			for(l = 0; l < 8; l++)
				_c[j+(i+b_n)*M+b_m*8+l] += t[b_m][b_n][l];
}


//Matmul v4 + using FMA operation on a sub-part of C and optimizing the cache usage
//No need for transposition in this version as most of the work is done in cache anyway

void matmul_v5(const float *A, const float *B, float *C, int M, int N, int K) 
{
	int i,j;
	
	if(M % ker_h != 0 || N % ker_w != 0)
	{
		printf("Error, mismatch between matrix size and kernel size");
		exit(EXIT_FAILURE);
	}

	#pragma omp parallel for schedule(dynamic, 1) private(i,j)
	for(j = 0; j < M; j += ker_h)
		for(i = 0; i < N; i += ker_w)
			kernel(A, B, C, M, N, K, i, j, 0, K);

}


//v5 + blocking decomposition to successively load sub parts of A and B
//into L2 and L1 cache for maximum reuse. Maximizing the achievable memory bandwith

void matmul_v6(const float *A, const float *B, float *C, int M, int N, int K) 
{
	int i,j;
	
	const int l3 = 64; //64 //Number of rows from A
	const int l2 = 120; //120 //Number of columns from B
	const int l1 = 240; //240 //Number of columns from A
	
	if(M % ker_h != 0 || N % ker_w != 0 || l2 % ker_w != 0 || l3 % ker_h != 0)
	{
		printf("Error, mismatch between matrix size and kernel size");
		exit(EXIT_FAILURE);
	}

	int i1, i2, i3;
	
	#pragma omp parallel for schedule(dynamic, 1) private(i2,i1,i,j)
	for(i3 = 0; i3 < M; i3 += l3)
		for(i2 = 0; i2 < N; i2 += l2)
			for(i1 = 0; i1 < K; i1 += l1)
				for(j = i3; j < fmin(i3+l3,M); j += ker_h)
					for(i = i2; i < fmin(i2+l2,N); i += ker_w)	
						kernel(A, B, C, M, N, K, i, j, i1, fmin(i1+l1,K));
}


int main(int argc, char *argv[])
{
	
	size_t i;
	
	int l_size, M, N, K;
	float *A, *B, *C, *C_blas;

	if (argc == 2)
		l_size = atoi(argv[1]);
	else
		l_size = 1920;
	
	M = l_size; N = l_size; K = l_size;
	
	create_matrices(&A, &B, &C, &C_blas, M, N, K);
	
	//init_timing(&timer);

	//matmul_blas(A,B,C_blas,M,N,K);

	//printf("%f\n", elapsed_time(timer));

	init_timing(&timer);

	matmul_v6(A,B,C,M,N,K);
	
	printf("%f\n", elapsed_time(timer));
	
	
	//Uncomment this part when implementing a new version to verify the content of C against the OpenBLAS matmul
	
	/*
	for (i = 0; i < M*N; i++)
		if((C[i] - C_blas[i])*(C[i] - C_blas[i]) > 0.00001)
		{
			printf("ERROR ! MATRIX DIFF !\n");
			printf("i:%ld C:%f C_blas:%f\n", i, C[i], C_blas[i]);
			exit(EXIT_FAILURE);
		}
	*/
	exit(EXIT_SUCCESS);
}

		
		
