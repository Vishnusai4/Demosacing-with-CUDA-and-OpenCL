#!/usr/bin/env python

from pycuda import gpuarray
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

import time
import numpy as np 
import skimage.measure


class MaxPool:
	def __init__(self, mat, stride):

		self.matrix = mat
		self.stride = stride

		self.blockSize = None
		self.blockDim = None
		self.gridDim = None

		maxpool_kernel_naive = """
		__global__ void maxpool_naive(float *A, float *B, float *res, 
									const unsigned int stride)
		{

		}
		"""

		self.maxpool_func = SourceModule().get_function("maxpool_naive")

	def max_pool_python(self):
		return skimage.measure.block_reduce(self.mat, (self.stride, self.stride), np.max)

	def max_pool_naive(self):
		e_start = cuda.Event()
		e_end = cuda.Event()


if __name__ == '__main__':
	mat = np.array(
		[1, 2, 5, 6], 
		[3, 4, 7, 8], 
		[9, 10, 13, 14],
		[11, 12, 15, 16]
		)
	MaxPoolClass = MaxPool(mat)

	result_py = MaxPoolClass.max_pool_python()
	print("Max Pool Python: \n", result_py)

