import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import sys
import time

	

mod = SourceModule("""
	__global__ void piCalc(double *x, double *y, double *count, int* N) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x; // Thread id

	int threadCount = gridDim.x * blockDim.x;

	int countPointsInCircle = 0;
	for (int i = idx; i < N[0]; i += threadCount) {
		if (x[i] * x[i] + y[i] * y[i] < 1) {
			countPointsInCircle++;
		}
	}
	atomicAdd(count, countPointsInCircle);	

}

""")

args = int(sys.argv[1])

N = np.array([args])
N_c = N[0]

x = np.random.random(N_c)
y = np.random.random(N_c)
count = np.zeros(1)
z = np.zeros(N_c)

block_size = (512, 1, 1)
grid_size = (int(N_c / (128 * block_size[0])), 1)

pi_calc = mod.get_function("piCalc")

start = time.time()
pi_calc(drv.In(x), drv.In(y), drv.Out(count), drv.In(N), drv.Out(z), block = block_size, grid = grid_size)
drv.Context.synchronize()
end = time.time()

print(count * 4 / N_c)
