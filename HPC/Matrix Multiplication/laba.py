import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import sys
import time
mod = SourceModule("""
	__global__ void multiply(double* a, double* b, double* c, int* N){
		const int row = blockIdx.y * blockDim.y + threadIdx.y;
		const int column = blockIdx.x * blockDim.x + threadIdx.x;
		for(int i = 0; i < N[0]; i++){
			c[row * N[0] + column] += a[row * N[0] + i] * b[i * N[0] + column];		
		}	
	}
""")
args = int(sys.argv[1])

N = np.array([args])
N_c = N[0]

a = np.random.randn(N_c, N_c)
b = np.random.randn(N_c, N_c)
c = np.zeros((N_c, N_c))

block_size = (2, 2, 1)
grid_size = (int((N_c + block_size[0] - 1) / 2), int((N_c + block_size[1] - 1) / 2))

multiply = mod.get_function("multiply")

start = time.time()
multiply(drv.In(a), drv.In(b), drv.Out(c), drv.In(N), block = block_size, grid = grid_size)
drv.Context.synchronize()
end = time.time()

start_cpu = time.time()
c_cpu = np.zeros((N_c, N_c))
for i in range(N_c):
	for j in range(N_c):
		for k in range(N_c):
			c_cpu[i, j] += a[i, k] * b[k, j]
end_cpu = time.time()
print('Time of GPU {}'.format(end - start))
print('Time of CPU {}'.format(end_cpu - start_cpu))
if np.allclose(c, c_cpu, 0.1, 0.1):
	print('Matrixes from cpu and gpu are close')
else:
	print('Matrixes from cpu and gpu are different')
