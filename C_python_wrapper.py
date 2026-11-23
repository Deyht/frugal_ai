import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np


l_size = 512

if os.system("gcc matmul.c -o matmul -lm -lopenblas"): sys.exit("Compilation error")

c_code_output = os.popen("./matmul %d"%(l_size)).read()

if "ERROR ! MATRIX DIFF !" in c_code_output:
    print(c_code_output)
    exit()

openblas_time, custom_time = [float(line.split()[-2]) for line in c_code_output.splitlines()]

print ("OpenBLAS at size %d \t time %f s"%(l_size, openblas_time))
print ("Custom at size %d \t time %f s"%(l_size, custom_time))

#Use this wrapper to create loops over l_size for the different versions of matmul in C and plot the FLOPS curves



