import numpy as np 
from matplotlib.image import imread
import matplotlib.pyplot as plt

color = {"red": 0, "green": 1, "blue": 2}

# Bilinear Interpolation Demosaicing
def demosaic_bi(im):

    '''
    m[x][y][0] --> RED
    m[x][y][1] --> GREEN
    m[x][y][2] --> BLUE
    '''

    m = np.zeros((im.shape[0], im.shape[1], 3))
    shape = im.shape

    # Padded Image - Padded with '-1'
    im_p = np.negative(np.ones((im.shape[0] + 2, im.shape[1] + 2)))
    im_p[1:shape[0]+1,1:shape[1]+1] = im
    
    for y in range(1, im.shape[0]+1):
        for x in range(1, im.shape[1]+1):

            # Normalize to compensate for padding
            row = y-1
            col = x-1 

            # In this format:
            # X   X
            #   P
            # X   X
            corner_pixels = np.array([im_p[y-1][x-1], im_p[y-1][x+1], im_p[y+1][x-1], im_p[y+1][x+1]])
            corner_pixels = corner_pixels[corner_pixels >= 0]
            # In this format:
            #   X 
            # X P X
            #   X               
            edge_pixels = np.array([im_p[y][x-1], im_p[y][x+1], im_p[y-1][x], im_p[y+1][x]])
            edge_pixels = edge_pixels[edge_pixels >= 0]

            # In this format:
            # X P X
            l_r_pixels = np.array([im_p[y][x-1], im_p[y][x+1]])
            l_r_pixels = l_r_pixels[l_r_pixels >= 0]

            # In this format:
            #   X
            #   P
            #   X
            t_b_pixels = np.array([im_p[y-1][x], im_p[y+1][x]])
            t_b_pixels = t_b_pixels[t_b_pixels >= 0]

            if (y%2 == 1): # RED GREEN pattern row
                if (x%2 == 1): # RED pixel

                    # Get red pixel - copy
                    m[row][col][0] = im_p[y][x]
                    # Get green pixel - take avg of surrounding non-negative green pixels
                    m[row][col][1] = edge_pixels.mean()
                    # Get blue pixel - take avg of surrounding non-negative blue pixels
                    m[row][col][2] = corner_pixels.mean()

                else: # GREEN pixel

                    # Get red pixel - take avg of surrounding non-negative red pixels
                    m[row][col][0] = l_r_pixels.mean()
                    # Get green pixel - copy
                    m[row][col][1] = im_p[y][x]
                    # Get blue pixel - take avg of surrounding non-negative blue pixels
                    m[row][col][2] = t_b_pixels.mean()

            else: # BLUE GREEN pattern row
                if (x%2 == 1): # GREEN pixel

                    m[row][col][0] = t_b_pixels.mean()
                    m[row][col][1] = im_p[y][x]
                    m[row][col][2] = l_r_pixels.mean()

                else: # BLUE pixel

                    m[row][col][0] = corner_pixels.mean()
                    m[row][col][1] = edge_pixels.mean()
                    m[row][col][2] = im_p[y][x]

    return m
