# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 11:07:42 2025

@author: alxdn
"""
import numpy as np

"""
Handle :
    -Stride Division by 0 + tuple
    -Kernel 3x4
"""

def new_dimensions(img, k, strides, padding):
    
    h = int(((img.shape[0] - k[0] + (2*padding)) / strides) +1)
    w = int(((img.shape[1] - k[1] + (2*padding)) / strides) +1)
    
    return (h,w)

def add_padding(img, padding):
    
    r, c = padding[0], padding[1]
    h, w = img.shape
    new_padded_img = np.zeros((h + r*2, w + c*2))
    new_padded_img[r : h + r, c : w + c] = img
    
    return new_padded_img

def convolve2d(img, kernel, padding, strides):
    
    k = kernel.shape
    
    padded_img = add_padding(img, padding)
    target_size = new_dimensions(padded_img, k, strides, padding=0)
    
    new_img = np.zeros(target_size)
    
    
    for i in range(0, target_size[0]):
        for j in range(0, target_size[1]):
            
            matrix = padded_img[i:i+k[0], j:j+k[1]]
            new_img[i,j] = np.sum(matrix * kernel)
            
    return new_img


