
import numpy as np
import time

def matmul_naive(A, B, C, M, N, K):
	# Add your own naive 3-loop implementation here
	return


m_size = 512
M = m_size
N = m_size
K = m_size

np.random.seed(0)

A = ((np.random.random((M,K))-0.5)*0.1).astype("float32")
B = ((np.random.random((K,N))-0.5)-0.1).astype("float32")
C_verif = A@B

C = np.zeros((M,N), dtype="float32")

t_start = time.time()

matmul_sum(A, B, C, M, N, K)

ellapsed_time = time.time() - t_start # In seconds

print(ellapsed_time)

if(np.sum((C - C_verif)**2) > 0.00001):
	print("Error: results do not match!")
	exit()
