import numpy as np 
from matplotlib.image import imread
import matplotlib.pyplot as plt
import demosaic_nn

images = ['balloons.jpg', 'candy.jpg', 'cat.jpg', 'ip.jpg', 
			'puppy.jpg', 'squirrel.jpg', 'tree.jpg']
mosaiced_images = []
gtruth_images = []

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

def show_image(img):
	im = plt.imshow(img)
	plt.show()

def save_image(img, name):
	im = plt.imshow(img)
	plt.savefig("output_img_python/" + name + ".jpg")

generate_input(mosaiced_images, gtruth_images)

nn = demosaic_nn.demosaic_nn(mosaiced_images[0])
save_image(nn, "nearest")

