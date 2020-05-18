import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import sys
import time
import cv2

args = sys.argv[1]

mod = SourceModule("""
	texture<unsigned int, 2, cudaReadModeElementType> tex;

__global__ void filter(unsigned int* result, const int M, const int N, const float sigma_d, const float sigma_r)
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
        result[i*N + j] = s / c;
    }


}
""")

path_image = args[0]
sigma_r = float(args[1])
sigma_d = float(args[2])

image = cv2.imread(path_image)

N = image.shape[0]
M = image.shape[1]
block_size = (2, 2, 1)
grid_size = (int((N + block_size[0] - 1) / 2), int((M + block_size[1] - 1) / 2))

result = np.zeros((N, M))
result_gpu = np.zeros((N, M))

filter = mod.get_function("filter")

filter(drv.Out(gpu_result), np.int32(M), np.int32(N), np.float32(sigma_d), np.float32(sigma_r), block = block_size, grid = grid_size)

cv2.imwrite('labaresult.png', result_gpu)

