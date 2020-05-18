import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import sys
import time
import cv2

args = sys.argv

def filter_cpu(image, sigma_r, sigma_d):
	def filter_element(i, j):
		c = 0
		s = 0
    		for k in range(i-1, i+2):
        		for l in range(j-1, j+2):
            			g = np.exp(-((k - i) ** 2 + (l - j) ** 2) / sigma_d ** 2)
            			i1 = image[k, l]/255
            			i2 = image[i, j]/255
            			r = np.exp(-((i1 - i2)*255) ** 2 / sigma_r ** 2)
            			c += g*r
            			s += g*r*image[k, l]
    		result = s / c
    		return result
	
	result = np.zeros(image.shape)
	for i in range(1, image.shape[0] - 1):
		for j in range(1, image.shape[1] - 1)
			result[i, j] = filter_pixel(i, j)
	return result

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

path_image = args[1]
sigma_r = float(args[2])
sigma_d = float(args[3])

image = cv2.imread(path_image)

N = image.shape[0]
M = image.shape[1]
block_size = (2, 2, 1)
grid_size = (int((N + block_size[0] - 1) / 2), int((M + block_size[1] - 1) / 2))

result = filter_cpu(image, sigma_r, sigma_d)
result_gpu = np.zeros((N, M))

filter = mod.get_function("filter")

filter(drv.Out(result_gpu), np.int32(M), np.int32(N), np.float32(sigma_d), np.float32(sigma_r), block = block_size, grid = grid_size)

cv2.imwrite('labaresult.png', result_gpu)
cv2.imwrite('labaresult_cpu.png', result)

