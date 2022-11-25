# -*- coding: utf-8 -*-

from math import cos, pi, sqrt
import numpy as np

from numpy import empty,arange,exp,real,imag,pi
from numpy.fft import rfft,irfft

def dct_2d(image, numberCoefficients=0):
    
    nc = numberCoefficients
    height = image.shape[0]
    width = image.shape[1]
    imageRow = np.zeros_like(image).astype(float)
    imageCol = np.zeros_like(image).astype(float)

    for h in range(height):
        imageRow[h, :] = dct_1d(image[h, :], nc)
    for w in range(width):
        imageCol[:, w] = dct_1d(imageRow[:, w], nc)

    return imageCol

def dct_1d(image, numberCoefficients=0):
    
    nc = numberCoefficients
    n = len(image)
    newImage= np.zeros_like(image).astype(float)

  
    for k in range(n):
        sum = 0
        for i in range(n):
            sum += image[i] * cos((pi * k * (i + 1/2)) / n)
        ck = sqrt(0.5) if k == 0 else 1
        newImage[k] = sqrt(2.0 / n) * ck * sum
        # newImage[k] = sum

    if nc > 0:
        newImage.sort()
        for i in range(nc, n):
            newImage[i] = 0

    return newImage


def dct(y): #Basic DCT build from numpy
    N = len(y)
    y2 = empty(2*N,float)
    y2[:N] = y[:]
    y2[N:] = y[::-1]

    c = rfft(y2)
    phi = exp(-1j*pi*arange(N)/(2*N))
    return real(phi*c[:N])


def dct2(y): #2D DCT bulid from numpy and using prvious DCT function
    M = y.shape[0]
    N = y.shape[1]
    a = empty([M,N],float)
    b = empty([M,N],float)

    for i in range(M):
        a[i,:] = dct(y[i,:])
    for j in range(N):
        b[:,j] = dct(a[:,j])

    return b
