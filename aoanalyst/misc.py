#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:32:36 2019

@author: aurelien
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 13:40:30 2017

@author: Aurelien

Contains a set of functions of general use.
"""
import os
from cv2 import imwrite
import numpy as np
import h5py
import time

from numpy.linalg import lstsq
from skimage.io import imsave
import glob
import sys
from shutil import copyfile

import aotools
from aotools.ext import czernike

experiment_class = ["modal","fcs","correlation","FCS_calibration"]

class AberrationNames(object):
    def __init__(self):
        self.names = ['High NA defocus','tip','tilt','defocus','Oblique astigmatism'\
                    ,'Vertical astigmatism','Vertical coma','Horizontal coma'\
                    ,'Vertical trefoil','Oblique trefoil','1st spherical',\
                    'Vertical secondary astigmatism',\
                    'Oblique scondary astigmatism',\
                    'Vertical quadrafoil','Oblique quadrafoil']
    def __getitem__(self,nb):
        if isinstance(nb,(int,np.integer)):
            if nb<0:
                return "Not a zernike mode"
            elif nb<len(self.names):
                return self.names[nb]
            elif nb==21:
                return "2nd spherical"
            elif nb==36:
                return "3rd spherical"
            else:
                return "mode "+str(nb)
            
        elif type(nb)==list or type(nb)==np.ndarray:
            out=[]
            for elt in nb:
                out.append(self.__getitem__ (elt) )
            return out
        
aberration_names = AberrationNames()

def extract_images(log_images,file,P,modes=None,number=None):
    """
    Method used to extract the images in h5 files and save them to make them 
    easier to observe
    Parameters: 
        log_images: np.ndarray, 3D array containing the images.
        file: string, name of the file
        P: int, number of measurement points per mode
        modes: array or list, indices of the Zernike modes. If None, considers 
        the N first modes, N being calculated from P and the shape of the data.
        number: int, experiment number. If None, does not assign a number to an experiment
        
        
    """
    if modes is None:
        Nz = log_images.shape[0]//P
        modes = np.arange(1,Nz+1,dtype = np.int)
    filename = file[:-3] + "_images"
    if number:
        filename = str(number)+"_"+filename
    if not os.path.isdir(filename):
        os.mkdir(filename)
    for j,mode in enumerate(modes):
        sub_images = log_images[j*P:(j+1)*P,:,:]
        path_mode = os.path.join(filename,aberration_names[mode])
        if not os.path.isdir(path_mode):
            os.mkdir(path_mode)
        for i in range(P):
            sub_sub_image = sub_images[i,:,:]
            imwrite(os.path.join(path_mode,str(i)+".png"), sub_sub_image)
            

def fancy_kurtosis(stack,mu=None,std=None,normalise=True):
    """
    Computes the fourth standardized moment of a stack of images. 
    Parameters:
        stack: np.ndarray of dimension 3: (number_of_images, width, height)
    Returns:
        k: np.ndarray, 1 dimensional: the fancy kurtosis of each image in the stack.
    """

    if mu is None:
        mean0 = np.mean(stack)
    else:
        mean0 = mu
    if std is None:
        std0 = np.std(stack)
    else:
        std0 = std
    f_k = lambda x: np.sum(((x-mean0)/std0)**4)
    
    if type(stack)==list:
        stack = np.asarray(stack)
    k = np.zeros(stack.shape[0])
    for i in range(k.size):
        im = stack[i,:,:]
        im[im<(mean0-std0)]=mean0-std0
        k[i] = f_k(im)
    k/=float(stack.size)
    if normalise:
        k /= np.max(k)
    return k

def entropy(image,m=None,M=None):
    """
    Computes the entropy of the histogram of an image
    Parameters:
        image: numpy 2D array
    Returns:
        entropy: float, tha value of the entropy
    """
    from scipy.ndimage.filters import gaussian_filter
    image = image.astype(np.float)
    #image = gaussian_filter(image,0.2)
    image = image/np.max(image)
    if m is None or M is None:
        hist,bin_edges = np.histogram(image,bins=100,density=False)
    else:
        hist,bin_edges = np.histogram(image,bins=100,range=(m,M),density=False)
    hist = hist[hist>0]
    hist = hist.astype(np.float)
    #hist/=np.sum(hist)
    """
    un = np.unique(image)
    hist = np.zeros(un.size)
    for i,val in enumerate(un):
        hist[i] = np.count_nonzero(image==val)
    assert(np.sum(hist)==image.size)"""
    hist /=np.sum(hist)
    entropy = -1*np.sum(hist*np.log(hist))
    return entropy

def apply_metric(image_list,fun):
    if type(image_list)!=list:
        image_list = image_list.tolist()
    out = np.array(list(map(fun,image_list)))
    return out/np.max(out)


def file_extractor(file, open_stacks=True):
    h5f = h5py.File(file, 'r')
    out = {} 
    for key in h5f['modal/'].keys():
        if key=="log_images" and not open_stacks:
            continue
        out[key] = h5f['modal/'][key][()]
    return out

def general_file_extractor(file):
    h5f = h5py.File(file, 'r')
    out = {} 
    for key in h5f.keys():
        if key in experiment_class:
            out["class"] = key
            for key2 in h5f[key]:
                out[key2] = h5f[key][key2].value
    return out

def comparison_file_extractor(file,open_stacks=True):
    """Opens the result of a comparison experiment.
    Parameters:
        file: str, path to file
        open_stacks: bool, if False does not load the stacks (for less memory consumption)"""
    out={}
    h5f = h5py.File(file, 'r')
    for k in h5f.keys():
        if k!="filenames":
            if k=="stacks" and not open_stacks:
                continue
            out[k] = h5f[k].value
        else:
            fn = {}
            for kk in h5f["filenames/"]:
                nr = int(kk[4:])
                fn[nr] = h5f["filenames"][kk].value
                print(kk,fn[nr])
            fn = sorted(fn.items())
            nrs = np.array([x[0] for x in fn])
            fn = [x[1] for x in fn]
            assert(np.all(nrs==np.arange(1,nrs.size+1)) )
            out["filenames"] = fn
    h5f.close()
    return out


def read_notes_cmp(file):
    """Reads the notes of a comparison experiment.
    Parameters:
        file: str, path to file
        """
    out={}
    h5f = h5py.File(file, 'r')
    out = h5f['notes'][()]
    h5f.close()
    return out


def modal_fcs_extractor(file):
    """Does not extract the stacks as they are useless in most cases and can
    be very large"""
    h5f = h5py.File(file, 'r')
    out = {} 
    for key in h5f['modal/'].keys():
        if key!="log_images":
            out[key] = h5f['modal/'][key].value
    return out

def zernike_analysis(phase, rho,n=4,n_modes=None):
    r"""Decompose the phase in Zernike modes using a least squares 
    regression"""

    # FIXME probably ENZ modes should be used instead of Zernike modes
    rzern = czernike.RZern(n)
    u,v = phase.shape
    x = np.linspace(-u/2,u/2,u)/rho
    y = np.linspace(-v/2,v/2,v)/rho
    xx,yy=np.meshgrid(x,y)
    rzern.make_cart_grid(xx,yy)
    zfm = np.isfinite(rzern.ZZ[:, 0])
    zfA = rzern.ZZ[zfm, :]
    if n_modes is not None:
        zfA = zfA[:,:n_modes]
    uph1 = phase.reshape(-1)[zfm]
    
    alpha_hat = lstsq(zfA, uph1)[0]

    return alpha_hat,rzern.ZZ

def zernike_generation(phase, rho,mode):
    r"""Generates a zernike phase mask with radius rho.
    Parameters:
        phase: numpy 2D array,phase mask used as a template
        rho: float,radius of the phase to generate
        mode: int, mode number
    Returns:
        zfA: 2D numpy array, phase mask of mode with amplitude 1rad rms"""

    # FIXME probably ENZ modes should be used instead of Zernike modes
    if mode<15:
        n=4
    elif mode<28:
        n=6
    else:
        n=8
    rzern = czernike.RZern(n)
    u,v = phase.shape
    x = np.linspace(-u/2,u/2,u)/rho
    y = np.linspace(-v/2,v/2,v)/rho
    xx,yy=np.meshgrid(x,y)
    rzern.make_cart_grid(xx,yy)
    zfA = rzern.ZZ[:, mode].reshape(u,v)
    zfA[~np.isfinite(zfA)] = 0
    return zfA

def array_zernike_generation(phase, rho,ab):
    r"""Generates a zernike phase mask with radius rho.
    Parameters:
        phase: numpy 2D array,phase mask used as a template
        rho: float,radius of the phase to generate
        mode: numpy 1D array, Zernike coefficients
    Returns:
        zfA: 2D numpy array, phase mask of mode with amplitude 1rad rms"""

    # FIXME probably ENZ modes should be used instead of Zernike modes
    if ab.size<=15:
        n=4
    elif ab.size<=28:
        n=6
    else:
        n=8
    rzern = czernike.RZern(n)
    u,v = phase.shape
    x = np.linspace(-u/2,u/2,u)/rho
    y = np.linspace(-v/2,v/2,v)/rho
    xx,yy=np.meshgrid(x,y)
    rzern.make_cart_grid(xx, yy)
    out=rzern.eval_grid(ab).reshape(phase.shape)
    return out

def svg_name_allocation(name=None):
    """Generates a file name to save the data of an experiment in a separate
    folder.
    Returns:
        h5fn: file name."""
    if not os.path.isdir("data"):
            os.mkdir("data")
    if not os.path.isdir("data/"+ time.strftime("%d_%m_%y")):
        os.mkdir("data/"+ time.strftime("%d_%m_%y"))
    path = "data/"+ time.strftime("%d_%m_%y/")
    
    h5fn = time.strftime("%d_%m_%y_%Hh%Mm%S.h5").replace("/","-")
    if name is not None:
        h5fn = name+h5fn
    h5fn =  path+h5fn
    return h5fn

def get_scan_parameters(active_measurement):
    """From an Imspector measurement, returns a dictionary containing the X,Y and
    Z resolution.
    Parameters: 
        active_measurement: Imspector-type measurement object
    Returns:
        scan_parameters: string, contains a dictionary with scan resolution and length"""
    Xres = active_measurement.parameters('ExpControl')["XRes"]  #N pixels along direction X
    XLen = active_measurement.parameters('ExpControl')["XLen"]
    
    Yres = active_measurement.parameters('ExpControl')["YRes"]  #N pixels along direction X
    YLen = active_measurement.parameters('ExpControl')["YLen"]
    
    Zres = active_measurement.parameters('ExpControl')["ZRes"]  #N pixels along direction X
    ZLen = active_measurement.parameters('ExpControl')["ZLen"]
    
    scan_resolution = [Xres,Yres,Zres]
    scan_length = [XLen,YLen,ZLen]
    scan_parameters = {"scan_resolution":scan_resolution,
                       "scan_length":scan_length}
    scan_parameters = str(scan_parameters)
    return scan_parameters

def isolate_correction_applied(folder):
    files = glob.glob(folder+"/*.h5")
    
    dst_folder=folder+"/wo_apply_corr"
    if not os.path.isdir(folder+"/wo_apply_corr"):
        os.mkdir(dst_folder) 
    for file in files:
        try:
            ext = file_extractor(file)
            
            if 'apply_corr' in ext:
                if not ext['apply_corr']:
                    fname = os.path.split(file)[-1]
                    copyfile(file,dst_folder+"/"+fname)
        except:
            pass

def find_optimal_arrangement(k):
    """Given a number of plots k, finds the best way to display them within a 
    rectangular area"""
    prime_numbers=[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    if k<=3:
        return 1,k
    if k in prime_numbers:
        k+=1
    estimator = int(np.ceil(np.sqrt(k))) #First try
    
    while estimator>=1 and k%estimator!=0:
        estimator+=1
    m = estimator
    n = k//estimator
    return n,m

def test_optimal_arrangement():
    import matplotlib.pyplot as plt
    ks = np.arange(3,28)
    js=1
    plt.figure()
    for k in ks:
        plt.subplot(5,5,js)
        js+=1
        m,n = find_optimal_arrangement(k)
        img = np.zeros(m*n)
        img[0:k] = np.arange(k)
        img = img.reshape(m,n)
        plt.imshow(img)
        plt.title(str(k))
        
def extract_correction_comparisons(folder):
    files = glob.glob(folder+"/*.h5")
    
    for file in files:
        out = comparison_file_extractor(file)
        filenames = out["filenames"]
        stacks = out["stacks"]
        print(np.min(stacks)-2**15)
        if np.min(stacks)>=2**15:
            stacks-=2**15
            print("supression offset")
        sizex = out["psize_x"]
        sizez = out["psize_z"]
        notes = out["notes"]
        names =[ "".join(f.split(".")[:-1]) for f in filenames]
        
        fname = "".join(file.split(".")[:-1])+"/"
        if not os.path.isdir(fname):
            os.mkdir(fname)
        for j in range(stacks.shape[0]):
            
            nn = fname+str(j)+"_"+"".join(os.path.split(names[j])[-1])+str(sizex)+"x"+str(sizez)+"nm"+".tif"
            imsave(nn,stacks[j])
            with open(fname+"notes.txt","w") as f:
                f.write(notes)
                
def crop(img,frac=0.8):
    u,v = img.shape
    nu=int(u*frac/2)
    nv=int(v*frac/2)
    out = img[u//2-nu:u//2+nu,v//2-nv:v//2+nv].copy()
    return out
               
def prompter_pixel_size():
    
    psize = input("Please specify the pixel size (in nm):\n")
    
    r = input("Are you sure this value is correct? (yes/no/edit)")
    while r!="yes" and r!="no" and r!="edit":
        print("Please enter \"yes\" or \"no\" ")
        r = input("Are you sure these are correct? (yes/no)")
    if r=="edit":
        prompter_pixel_size()
    if r=="no":
        sys.exit(0)
    return int(psize)

def inverse_zernike_matrix(df_mat):
    """Calculates the inverse of a "SLM calibration" matrix, that is supposed to be
    upper triangular"""
    I = np.eye(df_mat.shape[0])
    N = df_mat-I
    idf_mat = I-N
    assert( (np.dot(df_mat,idf_mat)==I).all())
    assert( (np.dot(idf_mat,df_mat)==I).all())
    return idf_mat

        