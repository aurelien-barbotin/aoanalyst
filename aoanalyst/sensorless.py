#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:33:23 2019

@author: aurelien
"""

import numpy as np
import pywt

from scipy import ndimage
from aoanalyst import msvst
from aoanalyst.misc import crop

def wavelet_decomposition(image,wlt='sym2'):
    """Performs the wavelet decomposition of an image using the pyWavelets package.
    Parameters:
        image: numpy array, 2D.
        wlt: string, optional. Specifies the wavelet to use for the decomposition.
    Returns:
        images: list, contains the different wavelets coefficients. With pyWavelets
        version 0.5.2, the first coefficient images[0] is the one with the highest
        resolution.
    """
    lvl=4
    #Reshaping into the correct multiple of 2
    new_size = (image.shape[0]//2**lvl)*2**lvl,(image.shape[1]//2**lvl)*2**lvl
    image = image[0:new_size[0],0:new_size[1]]
    coeffs_trous = pywt.swt2(image,wlt,lvl,start_level=0)
    images=[]
    for elts in coeffs_trous:
        cA,(cH,cV,cD) = elts
        images.append(cA)
    return images

def wlt_vector(image,lvl=None):
    wlts = wavelet_decomposition(image)
    if lvl is not None:
        wlts = wlts[0:lvl]
    coeffs = np.asarray(list(map(lambda x: np.sum(x**2) ,wlts)))
    coeffs/=np.sqrt(np.sum(coeffs**2))
    return coeffs

def fitter_exp(x, x0, a, b, c):
    return a*np.exp(-b*((x - x0)**2)) + c

def wavelet_metric(array,coeff=-1):
    """Computes a wavelet-based image quality metric, for an image or a stack of
    images. The principle is the following:
        1/Compute the 4-levels wavelet decomposition of the image
        2/Calculates the energy (sum of squared intensities) of each coefficient
        3/Normalises this 4-dimensional vector using its L2 norm.
        4/Picks one of the coefficients, typically the first or last one
    Parameters:
        array: numpy array, 2- or 3-dimensional. If 3D, array is a stack of images.
        coeff: int, optional. The index of the coefficient used for aberration 
        sensing.
    Returns:
        y: float or list of floats, the metric value for the image or stack.
    """
    
    ndims = array.ndim
    assert(ndims==2 or ndims==3)
    
    #Helper function which computes the metric value for one image

    if ndims==2:
        coeffs = wlt_vector(array)
        if coeff ==-1:
            y = 1-coeffs[coeff]
        else:
            y = coeffs[coeff]
    elif ndims==3:
        n=array.shape[0]
        y=[]
        for i in range(n):
            vec = wlt_vector(array[i,:,:])
            if coeff==-1:
                y.append(1-vec[coeff] )
            else:
                y.append(vec[coeff] )
    return y


def sobel(im,norm=True):
    im = im.astype(np.float)
    vert = ndimage.sobel(im,axis=0)
    hori = ndimage.sobel(im,axis=1)
    out = np.sum((crop(vert)**2+crop(hori)**2))
    if norm:
        out/=np.sum(im**2)
    return out

def decompose_dwt(img,lvl=4):
    out = []
    u,v = img.shape
    u = u//2**lvl*2**lvl
    v = v//2**lvl*2**lvl
    img = img[0:u,0:v]
    for i in range(lvl):
        cA,cD = pywt.dwt2(img,"db1")
        out.append(cA)
        img = cA
    return out

def dwt_metric(img):
    dec = decompose_dwt(img)
    dec = [sobel(x*1.0) for x in dec]
    return np.array(dec)

from scipy.stats import entropy

def wavelet_last_normalise(img):
    ajs2, dt = msvst.dec(img.astype(np.float),J=5)
    dt = dt[1:]
    last = ajs2[-1]
    def E(psi):
        return(np.sum(np.abs(psi)))
    vect = np.asarray([E(x)/E(last) for x in dt])
    return vect

def wavelet_last_normalise_sq(img):
    ajs2, dt = msvst.dec(img.astype(np.float),J=5)
    dt = dt[1:]
    last = ajs2[-1]
    def E(psi):
        return(np.sum(psi**2))
    vect = np.asarray([E(x)/E(last) for x in dt])
    return vect

def wavelet_higher_momentum(img,normalise=False):
    ajs2, dt = msvst.dec(img.astype(np.float),J=5)
    def hm(im):
        imc = np.abs(im)
        histo = imc[imc>2*np.std(im)]
        return np.var(histo)
    dt = np.asarray([hm(crop(d) ) for d in dt ])
    if normalise:
        dt = dt/np.sum(dt**2)
    return dt


def entropy_metric(image):
    histo,osef = np.histogram(image.reshape(-1))
    histo = histo.astype(np.float)
    histo/=np.sum(histo)
    return entropy(histo)

def ratio(l):
    return l[0]/l[1]

def fcs_wavelet(data,sq=False,lvl=None):
    
    #data = fcs.sum_n(data,10)
    db1 = pywt.Wavelet('db1')
    if lvl is None:
        lvl = min(pywt.dwt_max_level(data.size, db1),5)
    
    wr= pywt.wavedec(data, db1, mode='constant', level=lvl)
    if sq:
        wr = np.array([np.sum(x**2 ) for x in wr[1:]])
        wr/=np.sqrt(np.sum(wr**2))
    else:
        wr = np.array([np.sum( np.abs(x) ) for x in wr[1:]])
        wr/=np.sum(wr)
    return wr



metrics = {
            "std":np.std,
            "normalised variance":lambda x: np.std(x/np.mean(x)),
            "Sum squared":lambda x:np.sum(x**2),
           "intensity":lambda x: float(np.sum(x))
           }