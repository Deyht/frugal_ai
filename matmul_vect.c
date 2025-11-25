
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


//v3 + manual vectorization through memory aligned SIMD operations 
//Note, require C11 ISO, not compatible with C99 while auto vectorization is compatible

// Define a memory aligned vector type of 32 Bytes -> 8 float of 4 bytes (32 bits) each
typedef float vec __attribute__ (( vector_size(32) ));

void matmul_v4(const float *A, const float *B, float *C, int M, int N, int K)
{
	if(K % 8 != 0)
	{
		printf("Error, mismatch between matrix size and kernel size");
		exit(EXIT_FAILURE);
	}
	
	//Dynamically allocate a space in memory composed of X elements using the size of one element as alignement
	//vec *a = (vec*) aligned_alloc(sizeof(vec),X*sizeof(vec));  
	
	//Access a vector sub element i
	// a[i/8][i%8] = A[i];
	
	//Allocate and initialize a vector to a constant value
	//vec acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	
	//1° Create the vectorized versions of A and B
	
	//2° Fill the vectorized versions
	
	//3° Do the i,j,k loops with the vectorized accumulate loop over k
	
}



//Matmul using FMA operation on a sub-part of C and optimizing the cache usage
//No need for transposition in this version as most of the work is done in cache anyway

#define ker_w 6
#define ker_h 16

void kernel(const float *_a, const float *_b, float *_c, 
	int M, int N, int K, int i, int j, int k_start, int k_stop) 
{
	int k, b_m, b_n, l;
	float val;
	//declared in cache
	vec t[ker_h/8][ker_w] = {};

	//1° Loop over k that do the vectorized accumulate inside the kernel region using an implicit fma instruction

	//2° accumulate the result in the corresponding region of C
}


void matmul_v5(const float *A, const float *B, float *C, int M, int N, int K) 
{
	int i,j,k;
	
	if(M % ker_h != 0 || N % ker_w != 0)
	{
		printf("Error, mismatch between matrix size and kernel size");
		exit(EXIT_FAILURE);
	}

	//Simple loop over M and N kernel regions that call the kernel

}



//V4 but with blocking decomposition to successively load sub parts of A and B
//into L2 and L1 cache for maximum reuse. Maximizing the achievable memory bandwith

void matmul_v6(const float *A, const float *B, float *C, int M, int N, int K) 
{
	int i,j,k;
	
	// Set the size of the blocks and order them by importance into the different cache levels
	const int l3 = 1; //Number of rows from A
	const int l2 = 1; //Number of columns from B
	const int l1 = 1; //Number of columns from A
	
	if(M % ker_h != 0 || N % ker_w != 0 || l2 % ker_w != 0 || l3 % ker_h != 0)
	{
		printf("Error, mismatch between matrix size and kernel size");
		exit(EXIT_FAILURE);
	}

	//5-nested loop version that goes through the blocks in cache and then call the kernel on the sub regions in the current block
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

	matmul_v4(A,B,C,M,N,K);
	
	printf("%f\n", elapsed_time(timer));
	
	/*
	//Uncomment this part when implementing a new version to verify the content of C against the OpenBLAS matmul
	for (i = 0; i < M*N; i++)
		if((C[i] - C_bis[i])*(C[i] - C_bis[i]) > 0.00001)
		{
			printf("ERROR ! MATRIX DIFF !\n");
			printf("i:%ld C:%f C_bis:%f\n", i, C[i], C_bis[i]);
			exit(EXIT_FAILURE);
		}
	*/
	exit(EXIT_SUCCESS);
}
		
		
