
#include <stdlib.h>
#include <stdio.h>
#include <cblas.h>

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
	// Add your hown naive 3-loop implementation here
}


int main(int argc, char *argv[])
{
	size_t i;
	
	int l_size, M, N, K;
	float *A, *B, *C, *C_blas;

	if (argc == 2)
		l_size = atoi(argv[1]);
	else
		l_size = 512;
	
	M = l_size; N = l_size; K = l_size;

	create_matrices(&A, &B, &C, &C_blas, M, N, K);

	init_timing(&timer);

	matmul_blas(A,B,C_blas,M,N,K);

	printf("%f\n", elapsed_time(timer));

	//init_timing(&timer);

	//matmul_v1(A,B,C,M,N,K);
	
	//printf("%f s\n", elapsed_time(timer));
	
	/*
	//Uncomment this part when implementing a new version to verify the content of C against the OpenBLAS matmul
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

		
		
