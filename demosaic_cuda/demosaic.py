#!/usr/bin/env python

import time
import numpy as np

from pycuda import gpuarray
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

import demosaic_python.demosaic_nn as nn 
import demosaic_python.demosaic_bi as bi 
import demosaic_python.demosaic_agb as agb 

class Demosaic:
	def __init__(self, im):

		kernel = """
		#include <stdio.h>
		__global__ void demosaicGPU(float *in, float *out,
		const unsigned int w, const unsigned int h) 
		{

		}
		"""

		self.prg = SourceModule(kernel).get_function("demosaicGPU")

	def demosaic_nn_CPU(self):
		start = time.time()
		out_im_nn = nn.demosaic_nn(self.im)
		end = time.time()

		return out_im_nn, end - start

	def demosaic_bi_CPU(self):
		start = time.time()
		out_im_bi = bi.demosaic_bi(self.im)
		end = time.time()

		return out_im_bi, end - start

	def demosaic_agb_CPU(self):
		start = time.time()
		out_im_agb = agb.demosaic_adaptive(self.im)
		end = time.time()

		return out_im_agb, end - start


def generate_mosaic(image):
	mosaic = image[:, :, 1].copy() # green
	mosaic[::2, ::2] = image[::2, ::2, 0] # red
	mosaic[1::2, 1::2] = image[1::2, 1::2, 2] # blue
	return mosaic

def generate_input(mosaiced_images, gtruth_images):
	for file in images:
		image = imread('../images/' + file)
		image = image / 255.0
		gtruth_images.append(image)
		mosaiced_images.append(generate_mosaic(image))


if __name__ == '__main__':

	images = ['balloons.jpg', 'candy.jpg', 'cat.jpg', 'ip.jpg', 
			'puppy.jpg', 'squirrel.jpg', 'tree.jpg']
	mosaiced_images = []
	gtruth_images = []

	generate_input(mosaiced_images, gtruth_images)

	im = mosaiced_images[0] # Change this to change picture

	Demosaic = Demosaic(im)




