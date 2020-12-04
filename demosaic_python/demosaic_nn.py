import numpy as np 
from matplotlib.image import imread
import matplotlib.pyplot as plt

def demosaic_nn(im):
	'''
	m[x][y][0] --> RED
	m[x][y][1] --> GREEN
	m[x][y][2] --> BLUE

	Window Shape
	 ____
	|R G |  --> red,          green_right
	|G B |      green_left,   blue
	------
	
	
	'''

	# 3D Image Channel: Demosaiced
	im_d = np.zeros((im.shape[0], im.shape[1], 3))
	
	print("Shape of the Image: ", im.shape)
	
	# Iterate in intervals of 2
	for y in range(0, (im.shape[0]), 2):
		for x in range(0, (im.shape[1]), 2):
			# Manually add the four pixels
			try:
				if (y != im.shape[0]-1 or x != im.shape[0]-1):
					# Prepare the values from 2D Image
					red           = im[y][x]
					green_top     = im[y][x+1]
					green_bottom  = im[y+1][x]
					blue          = im[y+1][x+1]

					# Left Top Pixel: Missing Green and Blue
					im_d[y][x][0] = red
					im_d[y][x][1] = green_top # Green (right top)
					im_d[y][x][2] = blue 

					# Right Top Pixel: Missing Red and Blue
					im_d[y][x+1][0] = red
					im_d[y][x+1][1] = green_top
					im_d[y][x+1][2] = blue

					# Third Pixel: Missing Red and Blue
					im_d[y+1][x][0] = red
					im_d[y+1][x][1] = green_bottom
					im_d[y+1][x][2] = blue

					# Fourth Pixel: Missing Red and Green
					im_d[y+1][x+1][0] = red
					im_d[y+1][x+1][1] = green_bottom
					im_d[y+1][x+1][2] = blue
			except:
				continue
   				
	return im_d
