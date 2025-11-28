import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.system("export OMP_NUM_THREADS=1")
os.system("export OPENBLAS_NUM_THREADS=1")
os.system("export MKL_NUM_THREADS=1")
os.system("export NUMEXPR_NUM_THREADS=1")

import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit

def perf_curve_fct(c_fct, start=128, end=1920, size_step=128):

	nb_steps = (end-start+size_step)//size_step
	gflops_per_size = np.zeros((nb_steps,2))

	M = start; N = start; K = start

	A = (np.random.random((M,K)).astype("float32")-0.5)*0.1
	B = (np.random.random((K,N)).astype("float32")-0.5)-0.1
	C = np.zeros((M,N), dtype="float32")

	#first non-measured call in case it need to be recompiled
	c_fct(A, B, C, M, N, K)
	print ("Benchmarking %s"%(c_fct.__name__))
	for i in tqdm(range(0,nb_steps)):

		l_size = start + i*size_step
		M = l_size; N = l_size; K = l_size

		A = (np.random.random((M,K)).astype("float32")-0.5)*0.1
		B = (np.random.random((K,N)).astype("float32")-0.5)-0.1
		C = np.zeros((M,N), dtype="float32")

		t_start = time.time()
		c_fct(A, B, C, M, N, K)

		elapsed_time = time.time() - t_start

		gflops_per_size[i,0] = l_size
		gflops_per_size[i,1] = l_size**3/1e9/elapsed_time

	np.savetxt("%s_gflops_curve.txt"%(c_fct.__name__), gflops_per_size)
	return gflops_per_size


def matmul_naive(A, B, C, M, N, K):
	for i in range(0,M):
		for j in range(0,N):
			for k in range(0,K):
				C[i,j] += A[i,k] * B[k,j]


def matmul_vectorized(A, B, C, M, N, K):
	for i in range(0,M):
		for j in range(0,N):
			C[i,j] = np.sum(A[i,:]*B[:,j])


@jit(nopython=True, cache=True, fastmath=False)
def compiled_matmul_naive(A, B, C, M, N, K):
	for i in range(0,M):
		for j in range(0,N):
			for k in range(0,K):
				C[i,j] += A[i,k] * B[k,j]


@jit(nopython=True, cache=True, fastmath=False)
def compiled_matmul_vectorized(A, B, C, M, N, K):
	for i in range(0,M):
		for j in range(0,N):
			C[i,j] = np.sum(A[i,:]*B[:,j])

def at_operator(A, B, C, M, N, K):
	C = A@B

def numpy_matmul(A, B, C, M, N, K):
	C = np.matmul(A,B)

def numpy_dot(A, B, C, M, N, K):
	C = np.dot(A,B)


if(1):
	perf_curve_fct(matmul_naive, start=256, end=512, size_step=256)
if(1):	
	perf_curve_fct(matmul_vectorized)
if(1):
	perf_curve_fct(compiled_matmul_naive)
if(1):
	perf_curve_fct(compiled_matmul_vectorized)
if(0):
	perf_curve_fct(at_operator , end=4096)
if(0):
	perf_curve_fct(numpy_matmul, end=4096)
if(0):
	perf_curve_fct(numpy_dot   , end=4096)

matmul_naive_gflops_curve 					= np.loadtxt("matmul_naive_gflops_curve.txt")
matmul_vectorized_gflops_curve   			= np.loadtxt("matmul_vectorized_gflops_curve.txt")
compiled_matmul_naive_gflops_curve 			= np.loadtxt("compiled_matmul_naive_gflops_curve.txt")
compiled_matmul_vectorized_gflops_curve 	= np.loadtxt("compiled_matmul_vectorized_gflops_curve.txt")
#at_operator_gflops_curve 					= np.loadtxt("at_operator_gflops_curve.txt")
#numpy_matmul_gflops_curve 					= np.loadtxt("numpy_matmul_gflops_curve.txt")
#numpy_dot_gflops_curve	 					= np.loadtxt("numpy_dot_gflops_curve.txt")


plt.plot(matmul_naive_gflops_curve[:,0]					, matmul_naive_gflops_curve[:,1]				, label="Naiv")
plt.plot(matmul_vectorized_gflops_curve[:,0]			, matmul_vectorized_gflops_curve[:,1]			, label="Vectorized")
plt.plot(compiled_matmul_naive_gflops_curve[:,0]		, compiled_matmul_naive_gflops_curve[:,1]		, label="Compiled Naiv")
plt.plot(compiled_matmul_vectorized_gflops_curve[:,0]	, compiled_matmul_vectorized_gflops_curve[:,1]	, label="Compiled Vectorized")
#plt.plot(at_operator_gflops_curve[:,0]					, at_operator_gflops_curve[:,1]					, label="@ operator")
#plt.plot(numpy_matmul_gflops_curve[:,0]					, numpy_matmul_gflops_curve[:,1]				, label="Numpy matmul")
#plt.plot(numpy_dot_gflops_curve[:,0]					, numpy_dot_gflops_curve[:,1]					, label="Numpy dot")
plt.legend()
plt.gca().set_xlabel("Size N")
plt.gca().set_ylabel("GFLOPS")
plt.savefig("perf_curves_python.png")




