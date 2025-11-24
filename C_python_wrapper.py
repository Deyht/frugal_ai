import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np


l_size = 1024

if os.system("gcc matmul.c -o matmul -lm -lopenblas"): sys.exit("Compilation error")

elapsed_time_sgemm = float(os.popen("./matmul %d"%(l_size)).read())

print ("SGEMM BLAS at size %d \t time %f s"%(l_size, elapsed_time_sgemm))


