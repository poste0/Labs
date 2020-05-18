import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import sys
import time

	

mod = SourceModule("""
	__global__ void piCalc(double *x, double *y, double *count, int* N) {
texture<unsigned int, 2, cudaReadModeElementType> tex;

__global__ void interpolate(unsigned int * __restrict__ d_result, const int M, const int N, const float sigma_d, const float sigma_r)
{
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;


    if ((i<M)&&(j<N)) {
        float s = 0;
        float c = 0;
        for (int l = i-1; l <= i+1; l++){
            for (int k = j-1; k <= j+1; k++){
                float img1 = tex2D(tex, k, l)/255;
                float img2 = tex2D(tex, i, j)/255;
                float g = exp(-(pow(k - i, 2) + pow(l - j, 2)) / pow(sigma_d, 2));
                float r = exp(-pow((img1 - img2)*255, 2) / pow(sigma_r, 2));
                c += g*r;
                s += g*r*tex2D(tex, k, l);
            }
        }
        d_result[i*N + j] = s / c;
    }


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
pi_calc(drv.In(x), drv.In(y), drv.Out(count), drv.In(N), block = block_size, grid = grid_size)
drv.Context.synchronize()
end = time.time()

count_cpu = 0
start_cpu = time.time()
for i in range(N_c):
	if x[i] ** 2 + y[i] ** 2 < 1:
		count_cpu += 1
end_cpu = time.time()
print('Time of GPU {}'.format(end - start))
print('Time of CPU {}'.format(end_cpu - start_cpu))
print('Pi of GPU {}'.format(count * 4 / N_c))
print('Pi of CPU {}'.format(count_cpu * 4 / N_c))
print('Time of GPU / Time of CPU {}'.format((end_cpu - start_cpu) / (end - start)))
print('Pi of CPU - Pi of GPU {}'.format((count_cpu - count) * 4 / N_c))
