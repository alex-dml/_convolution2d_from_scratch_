# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 15:13:28 2025

@author: alxdn
"""
from functions import convolve2d
import cv2
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    
    img = cv2.imread('lenna.png')
    img = cv2.resize(img, (512, 500))
    
    plt.imshow(img, cmap='gray')
    plt.title('Image before convolution')
    plt.show()
    
    kernel = np.array([[-1, -1, -1, -1],
                      [-1, 8, -1, -1],
                      [-1, -1, -1, -1]])
    
    r = convolve2d(img[:,:,0], kernel, padding=(1,3), strides=1)
    g = convolve2d(img[:,:,0], kernel, padding=(1,3), strides=1)
    b = convolve2d(img[:,:,0], kernel, padding=(1,3), strides=1)
    res = np.dstack((r,g,b))

    plt.imshow(res, cmap='gray')
    plt.title('Image after convolution')
    plt.show()
    
    print("Image size before convolution :", img.shape)
    print("Image size after convolution :", res.shape)
    