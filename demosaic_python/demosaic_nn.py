import numpy as np 
from matplotlib.image import imread
import matplotlib.pyplot as plt

def demosaic_nn(im):
	count = 0
	m = np.zeros((im.shape[0], im.shape[1], 3))
	
	print(im.shape)
	
	for y in range(0, (im.shape[0]), 2):
		for x in range(0, (im.shape[1]), 2):
			# Manually add the four pixels
			try:
				if (y != im.shape[0]-1 or x != im.shape[0]-1):
					m[y][x][0] = im[y][x]
					m[y][x][1] = im[y][x+1]
					m[y][x][2] = im[y+1][x+1]

					# Second Pixel: Missing Red and Blue
					m[y][x+1][1] = im[y][x+1]
					m[y][x+1][0] = im[y][x]
					m[y][x+1][2] = im[y+1][x+1]

					# Third Pixel: Missing Red and Blue
					m[y+1][x][1] = im[y+1][x]
					m[y+1][x][0] = im[y][x]
					m[y+1][x][2] = im[y+1][x+1]

					# Fourth Pixel: Missing Red and Green
					m[y+1][x+1][2] = im[y+1][x+1]
					m[y+1][x+1][0] = im[y][x]
					m[y+1][x+1][1] = im[y+1][x]
			except:
				continue
   				
	return m
