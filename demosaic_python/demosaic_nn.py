import numpy as np 
from matplotlib.image import imread
import matplotlib.pyplot as plt

RIGHT = "right"
RIGHT_CORNER = "right corner"
BOTTOM = "bottom"

def closest_pixel(r, c, region, im):
	# Function to simplify boundary cases
	# Ex: If we are looking at the last col, 
	#     and usually we copy the val from the pixel to the right
	#     it would instead copy the val from the left

	shape = im.shape

	if (region == RIGHT): # Example: For pixel [0, 0], the green pixel is on the right
		if (is_out_of_bounds(r, c+1, shape)):
			# Return the pixel to the left instead
			return im[r][c-1]
		else:
			return im[r][c+1]
	elif (region == RIGHT_CORNER): # Example: For pixel [0, 0]: the blue pixel is on the right corner
		if (is_out_of_bounds(r+1, c+1, shape)):
			# Return the left top pixel instead
			return im[r-1][c-1]
		else:
			return im[r+1][c+1]
	elif (region == BOTTOM): # Example: For pixel [0, 1]: the blue pixel is directly on the bottom of the green pixel
		if (is_out_of_bounds(r+1, c, shape)):
			return im[r-1][c]
		else:
			return im[r+1][c]
	else:
		print("Error: " + region + " command not recognized")
		return 0

def is_out_of_bounds(r, c, shape):
	return (r < 0 or r >= shape[0] or c < 0 or c >= shape[1])

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
	shape = im.shape
	
	print("Shape of the Image: ", im.shape)

	for r in range(0, im.shape[0]):
		for c in range(0, im.shape[1]):

			if (r % 2 == 0): # RED GREEN pattern row
				if (c % 2 == 0): # RED pixel
					im_d[r][c][0] = im[r][c]
					im_d[r][c][1] = closest_pixel(r, c, RIGHT, im)
					im_d[r][c][2] = closest_pixel(r, c, RIGHT_CORNER, im)
				else: # GREEN pixel
					im_d[r][c][0] = closest_pixel(r, c, RIGHT, im)
					im_d[r][c][1] = im[r][c]
					im_d[r][c][2] = closest_pixel(r, c, BOTTOM, im)
			else: # GREEN BLUE pattern row
				if (c % 2 == 0): # Green pixel
					im_d[r][c][0] = closest_pixel(r, c, BOTTOM, im)
					im_d[r][c][1] = im[r][c]
					im_d[r][c][2] = closest_pixel(r, c, RIGHT, im)
				else: # Blue pixel
					im_d[r][c][0] = closest_pixel(r, c, RIGHT_CORNER, im)
					im_d[r][c][1] = closest_pixel(r, c, RIGHT, im)
					im_d[r][c][2] = im[r][c]
				
	return im_d
