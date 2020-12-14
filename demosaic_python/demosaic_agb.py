import numpy as np 
from matplotlib.image import imread
import matplotlib.pyplot as plt

class Pixel:
	def __init__(self, x, y):
		self.x = x
		self.y = y

def get_pixel_value(im, l, r, t, b):
	# l -> left pixel, r -> right pixel, t -> top pixel, b -> bottom pixel

	shape = im.shape

	# Handling boundary conditions

	# Both pairs are valid: Non-edge pixels
	if (is_valid_pair(l, r, shape) and is_valid_pair(t, b, shape)):
		top_pixel = im[t.x][t.y]
		bottom_pixel = im[b.x][b.y]
		left_pixel = im[l.x][l.y]
		right_pixel = im[r.x][r.y]
		if (abs(top_pixel - bottom_pixel) > abs(left_pixel - right_pixel)):
			return (left_pixel + right_pixel) / 2
		else:
			return (top_pixel + bottom_pixel) / 2
	else: # At least one of the index pairs are out of bounds
		return average_of_valid_pixels(l, r, t, b, im)

def is_valid_pair(ind1, ind2, shape):
	return is_valid_pixel(ind1, shape) and is_valid_pixel(ind2, shape)

def is_valid_pixel(ind, shape):
	row_size = shape[0]
	col_size = shape[1]

	# If its out of bounds, return False
	if (ind.x < 0 or ind.x >= row_size or ind.y < 0 or ind.y >= col_size):
		return False
	
	# Pixel is NOT out of bounds
	return True

def average_of_valid_pixels(l, r, t, b, im):
	# If there are pixels that are out of the boundary, don't include them in the avg calculation

	row_size = im.shape[0]
	col_size = im.shape[1]

	pixels = [l, r, t, b]
	temp, count = 0, 0

	for p in pixels:
		if not (p.x < 0 or p.x >= row_size or p.y < 0 or p.y >= col_size):
			temp += im[p.x][p.y]
			count += 1

	if (count == 0): # This shouldn't happen because at least two of the pixels will be valid
		return 0
	return temp / count

def average_of_two(p1, p2, im):
	row_size = im.shape[0]
	col_size = im.shape[1]

	if (is_valid_pair(p1, p2, im.shape)):
		return (im[p1.x][p1.y] + im[p2.x][p2.y])/2
	elif(is_valid_pixel(p1, im.shape)):
		return im[p1.x][p1.y]
	else:
		return im[p2.x][p2.y]

# Adaptive Gradient Based Demosaicing
def demosaic_adaptive(im):
	'''
	m[x][y][0] --> RED
	m[x][y][1] --> GREEN
	m[x][y][2] --> BLUE
	'''
	
	m = np.zeros((im.shape[0], im.shape[1], 3))
	shape = im.shape
	row, col = shape[0], shape[1]

	for r in range(im.shape[0]):
		for c in range(im.shape[1]):

			left_top = Pixel(r-1, c-1)
			mid_top = Pixel(r-1, c)
			right_top = Pixel(r-1, c+1)
			left_mid = Pixel(r, c-1)
			right_mid = Pixel(r, c+1)
			bottom_left = Pixel(r+1, c-1)
			bottom_mid = Pixel(r+1, c)
			bottom_right = Pixel(r+1, c+1)

			if (r % 2 == 0): # RED GREEN pattern row
				if (c % 2 == 0): # RED pixel

					m[r][c][0] = im[r][c]
					m[r][c][1] = get_pixel_value(im, left_mid, right_mid, mid_top, bottom_mid)
					m[r][c][2] = get_pixel_value(im, left_top, bottom_right, bottom_left, right_top)

				else: # GREEN pixel

					m[r][c][0] = average_of_two(left_mid, right_mid, im)
					m[r][c][1] = im[r][c]
					m[r][c][2] = average_of_two(mid_top, bottom_mid, im)

			else: # GREEN BLUE pattern row
				if (c % 2 == 0): # GREEN pixel
					
					m[r][c][0] = average_of_two(mid_top, bottom_mid, im)
					m[r][c][1] = im[r][c]
					m[r][c][2] = average_of_two(left_mid, right_mid, im)

				else: # BLUE pixel

					m[r][c][0] = get_pixel_value(im, left_top, bottom_right, bottom_left, right_top)
					m[r][c][1] = get_pixel_value(im, left_mid, right_mid, mid_top, bottom_mid)
					m[r][c][2] = im[r][c]

	return m

