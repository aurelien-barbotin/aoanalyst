# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 14:18:39 2017

@author: Aurelien

This file contains miscalleneous functions useful to display data.
"""
import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from scipy.signal import convolve2d
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.axes_grid1 import make_axes_locatable

from skimage import transform as tf
from matplotlib_scalebar.scalebar import ScaleBar

from aoanalyst.io import file_extractor,comparison_file_extractor

from slm.slm import SLM
from scipy import ndimage

def sobelimage(im,horizontal = True,vertical=True):
    im = im.astype(np.float)
    u,v = im.shape
    
    frac = 0.06 #fraction to remove
    du = int(frac*u)
    dv = int(frac*v)
    im = im[du:u-du,dv:v-dv]
    hori=0
    vert=0
    if vertical:
        vert = ndimage.sobel(im,axis=0)
    if horizontal:
        hori = ndimage.sobel(im,axis=1)
    return (vert**2+hori**2)/np.sum(im)**2

def plot_details(xdata, ydata, images, optimum, ax=None, supp_values=None, 
                 zoom=None):
    """Displays a plot of aberration correction, and associates the extremum datapoints 
    with the corresponding image.
    Parameters:
        xdata: numpy array of the bias induced in radians
        ydata: numpy array of the metric values
        images: list of images
        optimum: float, optimal bias as determined by the optimisation algorithm
        ax: handle to plot. If None, creates a plot itself.
        supp_values: list of arrays with same shape as ydata containing data to be plotted
    """
    if not ax:
        fig, ax = plt.subplots()

    # Define a 2nd position to annotate (don't display with a marker this time)
    ax.plot(xdata,ydata,marker="o")
    if supp_values:
        for values in supp_values:
            ax.plot(xdata,values,marker="o")
    #vmin = np.min(images)
    def annotation_image(image_nr,zoom):
        offset = 70
        arr_img = images[image_nr]

        if zoom is None:
            zoom = 0.7
        imagebox = OffsetImage(arr_img, zoom=zoom,cmap="hot")
        imagebox.image.axes = ax
        #loc = 'upper left'
        y = ydata[image_nr]
        sgn = np.sign(0.5-y)
        if ydata.ndim>=2:
            y = ydata[image_nr,0]
        else:
            y=ydata[image_nr]
        ab = AnnotationBbox(imagebox, [xdata[image_nr],y],
                            xybox=(0,offset * sgn),
                            xycoords='data',
                            boxcoords="offset points",
                            pad=0.5,
                            arrowprops=dict(
                                arrowstyle="->",
                                connectionstyle="angle,angleA=0,angleB=90,rad=3")
                            )
        return ab
    ab1 = annotation_image(0,zoom)
    ab2 = annotation_image(-1,zoom)
    ax.add_artist(ab1)
    ax.add_artist(ab2)

    dists = np.abs([np.abs(x-optimum) for x in xdata])
    opt_pos = np.where(dists==np.min(dists))[0][0]  #If 2 solutions, we choose the first
    ab3 = annotation_image(opt_pos,zoom)
    ax.add_artist(ab3)
    # Fix the display limits to see everything
    #ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return ax


  
def interactive_plot(xdata,ydata,images,supp_values=None,fig=None):
    """
    Displays a plot of xdata, ydata and displays the image corresponding to 
    this point when hovering over data.
    Parameters:
        xdata: array, sequence of scalar
        ydata: array, metric values
        image: 3D array, of dimensions (n_images,width,height), containing the images
        from which the values ydata are calculated.
        supp_values: list of arrays, values of a different metric
    """
    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(111)
    line, = ax.plot(xdata,ydata, ls="-", marker="o")
    if supp_values is not None: 
        for ys in supp_values:
            ax.plot(xdata,ys)
            
    # create the annotations box
    if type(images)==list:
        images=np.asarray(images)
    im = OffsetImage(images[0,:,:], zoom=200/(images.shape[1]+images.shape[0]))
    xybox=(50., 50.)
    ab = AnnotationBbox(im, (0,0), xybox=xybox, xycoords='data',
            boxcoords="offset points",  pad=0.3,  arrowprops=dict(arrowstyle="->"))
    # add it to the axes and make it invisible
    ax.add_artist(ab)
    ab.set_visible(False)
    
    def hover(event):
        try:
        # if the mouse is over the scatter points
            if line.contains(event)[0]:
                # find out the index within the array from the event
                ind, = line.contains(event)[1]["ind"]
                # get the figure size
                w,h = fig.get_size_inches()*fig.dpi
                ws = (event.x > w/2.)*-1 + (event.x <= w/2.) 
                hs = (event.y > h/2.)*-1 + (event.y <= h/2.)
                # if event occurs in the top or right quadrant of the figure,
                # change the annotation box position relative to mouse.
                ab.xybox = (xybox[0]*ws, xybox[1]*hs)
                # make annotation box visible
                ab.set_visible(True)
                # place it at the position of the hovered scatter point
                ab.xy =(xdata[ind], ydata[ind])
                # set the image corresponding to that point
                im.set_data(images[ind,:,:])
            else:
                #if the mouse is not over a scatter point
                ab.set_visible(False)
        except Exception as e:
            print("Hovering error")
            print(e)
        fig.canvas.draw_idle()
    
    # add callback for mouse moves
    fig.canvas.mpl_connect('motion_notify_event', hover)
    return fig,ax

def scale_metric(met1,met2):
    """Sets the scale of metric 1 to metric 2 to simplify common visualisation"""
    if type(met2)==list:
        met2 = np.array(met2)
        met2 = met2.astype(np.float)
    if type(met1)==list:
        met1=np.array(met1)
        met1 = met1.astype(np.float)
    #Case when met2 is 2 dimensional like n wavelets coefficients.
    met2 = met2.squeeze()
    if met2.ndim==2:
        new_met2 = np.zeros_like(met2)
        for j in range(met2.shape[1]):
            new_met2[:,j] = scale_metric(met1,met2[:,j])
        return new_met2
    
    m = np.min(met1)
    M = np.max(met1)
    if M>m:
        met2-=np.min(met2)
        met2/=np.max(met2)
        met2 = met2*(M-m) + m
    return met2

def interactive_mode(file,mode,fig=None,supp_metrics=[],names=[]):
    """Displays an interactive plot for a certain mode in a file"""
    ext = file_extractor(file)
    xdata = ext['xdata'][:,mode]
    ydata = ext['ydata'][:,mode]
    P = ext["P"]
    images = ext["log_images"][P*mode:P*(mode+1)]
    
    supp_values=[]
    
    if supp_metrics is not None:
            for l,met in enumerate(supp_metrics):
                out=[]
                for j in range(P):
                    out.append(met(images[j]))
                out=scale_metric(ydata,out)
                supp_values.append(out)
    fig,ax=interactive_plot(xdata,ydata,images,fig=fig,supp_values=supp_values)
    legend=["Measured"]
    for nm in names:
        legend.append(nm)
    ax.legend(legend)

def overlay_mask2image(im1,im2):
    print("!!! overlay function not implemented")
    return im1
def image_comparison(im1,im2,fig=None):
    if fig is None:
        fig = plt.figure()
        
    ax1 = fig.add_subplot(321)
    imax1=ax1.imshow(im1)
    ax1.set_title("Image 1")
    ax1.set_axis_off()
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(imax1, cax=cax)
    
    ax2 = fig.add_subplot(322)
    ax2.set_title("Image 2")
    imax2 = ax2.imshow(im2)
    ax2.set_axis_off()
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(imax2, cax=cax)
    
    ax3 = fig.add_subplot(323)
    imax3=ax3.imshow(im1-im2)
    ax3.set_title("Difference Image1-Image2")
    ax3.set_axis_off()
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(imax3, cax=cax)
    
    diff = (im1-im2)>0
    over=overlay_mask2image(diff,im1)
    ax4 = fig.add_subplot(324)
    ax4.imshow(over )
    ax4.set_title("Difference>0")
    ax4.set_axis_off()
    
    normdiff = (im1/np.max(im1)-im2/np.max(im2))
    over=overlay_mask2image(diff,im1)
    ax5 = fig.add_subplot(325)
    imax5=ax5.imshow(normdiff )
    ax5.set_title("Difference of normalised Images")
    ax5.set_axis_off()
    divider = make_axes_locatable(ax5)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(imax5, cax=cax)

    
    over2=overlay_mask2image(normdiff>0,im1)
    ax6 = fig.add_subplot(326)
    ax6.imshow(over2 )
    ax6.set_title("Normalised Difference>0")
    ax6.set_axis_off()    
    
def plot_aberration_map(file,mode,fig = None,supp_metrics=None,fitter=None):
    ext = file_extractor(file)
    P = ext["P"]
    if "xdata" in ext:
        xdata = ext['xdata']    
        ydata = ext['ydata'].transpose()[mode].transpose()
        ydata/=np.max(ydata,axis=0)
    else:
        xdata=np.arange(P)
        ydata=np.ones(P)
   

    images = ext["log_images"][P*mode:P*(mode+1)]
    ksize=10        
    conv = np.ones((ksize,ksize))
    xdata = xdata[:,0]
    images_sobeled = []
    for i in range(images.shape[0]):
        sim=sobelimage(images[i])
        sim=convolve2d(sim,conv,mode='same',boundary='symm')
        images_sobeled.append(sim)

    outim = np.zeros_like(images_sobeled[0])
    for i in range(outim.shape[0]):
        for j in range(outim.shape[1]):
            yd=[]
            for k in range(P):
                yd.append(images_sobeled[k][i,j])
            yd = np.asarray(yd)
            if fitter is not None:
                popt,xh,yh=fitter.fit(xdata,yd)
                outim[i,j]=popt[0]
    
    """if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(1,2,1)
    ax.imshow(sobelimage(images[P//2]))
    ax.set_title('Sobel transform of image')

    ax2=fig.add_subplot(1,2,2)    
    ax2.imshow(outim)
    ax2.set_title('aberration map')"""
    image_list=[images[P//2],sobelimage(images[P//2]),outim]
    colorbar_plot(image_list,
                  titles=["Original Image","Sobel filter","Aberration map"],
                  fig=fig,cmap='viridis')
    
def plot_with_images(file,mode, fig = None, supp_metrics = None, show_exp = True):
    """Displays the result of a modal experiment. Displays each image below the
    corresponding datapoint
    Parameters:
        file: str, path to modal experiment
        mode: int, the index of the mode of interest (from 0 to #modes-1)
        fig: optional, if not None plots the results in fig
        supp_metrics: optional, list of image quality metrics (functions) to also
        display
        show_exp: bool, if True displays the quality metric measured during the
        experiment"""
    ext = file_extractor(file)
    if "log_images" not in ext:
        import h5py
        h5f = h5py.File(file, 'r')
        images = h5f['stacks']['stacks'][()]
        
        before = images[0]
        after = images[-1]
        log_images = images[1:-1]
    else:
        log_images = ext["log_images"]
    if "xdata" in ext:
        xdata = ext['xdata'][mode]
        ydata = ext['ydata'][mode]
        
        P = xdata.size
        ydata/=np.max(ydata,axis=0)
    else:
        xdata=np.arange(P)
        ydata=np.ones(P)
   

    images = log_images[P*mode:P*(mode+1)]
    
    if fig is None:
        fig = plt.figure()
    
    xoffset = 0.15
    yoffset = 0.15
    xsize = 0.75
    ysize = 0.60
    
    axplot = fig.add_axes([xoffset,yoffset,xsize,ysize])
    if show_exp:
        axplot.plot(xdata,ydata,marker="o")
    
    supp_values=[]
    if supp_metrics is not None:
            for l,met in enumerate(supp_metrics):
                out=[]
                for j in range(P):
                    out.append(met(images[j]))
                out=scale_metric(ydata,out)
                supp_values.append(out)
                axplot.plot(xdata,out,"-x")
                
    sfactor = xsize/P
    vmin=np.min(images)
    vmax=np.max(images)
    for k in range(P):
        axicon = fig.add_axes([xoffset+sfactor*k,0.8,sfactor,sfactor])
        axicon.imshow(images[k],cmap="hot",interpolation='nearest',picker=True,vmin=vmin,vmax=vmax)
        axicon.set_xticks([])
        axicon.set_yticks([])
    return axplot
    #image_comparison(images[0],images[P//2])

def mode_plot(file,mode=-1,metric=None):
    """Plots an interactive plot for mode in file"""
    values = file_extractor(file)
    try:
        P = values['P']
        bias = values['bias']
        images = values['log_images']
        modes = values['modes']
    except:
        raise IndexError("The file does not contain the right values") 
    if metric==None:
        metric= lambda x: kurtosis(x,axis=None)
    if mode!=-1:
        
        try:
            index = np.where(mode==modes)[0]
        except:
            raise IndexError('The mode {:d} has not been measured in this experiment'.format(mode))
    else:
        index = 0
    sub_images = images[index*P:(index+1)*P,:,:]
    yvals = list(map(metric,sub_images))
    xvals = np.linspace(-bias,bias,P)
    interactive_plot(xvals,yvals,sub_images)

def plot_n(elements,titles=None,cmap=None):
    """Plots n elements from a list using multiple subplots in a square layout.
    Parameters:
        elements: list, contains either images or a list of 1D array-like variables.
        Example: elements = [x,y1,y2]."""
    plt.figure()
    n=len(elements)
    xx  = math.ceil(np.sqrt(n))
    if cmap is None:
        cmap = "viridis"
    if type(elements[0])==np.ndarray and len(elements[0].shape)>=2:
        if n>3:
            for j in range(n):
                plt.subplot(xx,xx,j+1)
                plt.imshow(elements[j],cmap=cmap)
                if titles is not None:
                    plt.title(titles[j])
                plt.axis("off")
                plt.colorbar()
        else:
            for j in range(n):
                plt.subplot(1,n,j+1)
                plt.imshow(elements[j])
                if titles is not None:
                    plt.title(titles[j])
                plt.axis("off")
    else:
        
        for j in range(n):
            plt.subplot(xx,xx,j+1)
            #for ky in elements[j][1]:
            ky = elements[j][1]
            plt.plot(elements[j][0],np.asarray(ky)/np.max(ky))
            if titles:
                plt.title(titles[j])
                
def mid_plots(image):
    u,v=image.shape
    xplt = image[u//2,:]
    yplt=image[:,v//2]
    plt.figure()
    plt.subplot(221)
    plt.imshow(image)
    plt.axvline(u//2)
    plt.axhline(v//2)
    plt.title("Image")
    plt.subplot(222)
    plt.plot(xplt)
    plt.title("x profile")
    plt.subplot(223)
    plt.plot(yplt)
    plt.title("y profile")

def colorbar_plot(images_list,titles=None,nrows=1,scalebarval=None,fig=None,
                  cmap=plt.cm.gray,minval=None,maxval=None,scalefrees_list=[],
                  units = "nm",axes = None, show_colorbar = True,cbarpos="right"):
    """Displays a plot with colorbars to easily compare STED images.
    Parameters:
        images_list: list of images
        titles: optional, list of strings corresponding to the names of the images
        nrows: optional, number of rows over which images are displayed.
        scalebarval: if not None, should eb the pixel size in nm.
        fig: maptlotlib object in which the plot is plotted
        cmap: colormap
        minval: minimum value of the colorscale
        maxval: maximum value of the colorscale
        scalefrees_list: list of indices of images that should not be scaled by
        minval and maxval
        unit: str, name of units to be displayed on the scalebar
        axes: list, contains the axes where data is to be plotted
        show_colorbar: bool, if False does not display the colorbar. Ironic 
            for a colorbar plot
    Returns:
        fig: matplotlib object, handle to the figure
        axes: list of axes in fig
        """
    if titles is not None:
        assert(len(titles)==len(images_list))
    
    if type(images_list)!=list:
        images_list = [images_list]

    nn=len(images_list)
    nn = nn//nrows
    if fig is None:
        if axes is None:
            fig, axes = plt.subplots(ncols=nn,
                                     nrows=nrows,
                                       sharex=False,
                                       sharey=True,
                                       subplot_kw={"adjustable": "box-forced"})
            if type(axes)!=np.ndarray:
                axes=np.array([axes])
            axes = np.ravel(axes)
    else:
        axes=[]
        for i in range(nn):
            axes.append(fig.add_subplot(nrows,nn,i+1))
        
    for j,ax in enumerate(axes):
        try:#valid colormap
            if j in scalefrees_list:
                img0 = ax.imshow(images_list[j], cmap=cmap)
            else:
                img0 = ax.imshow(images_list[j], cmap=cmap,vmin=minval,vmax=maxval)
        except:
            cmap = "hot"
            if j in scalefrees_list:
                img0 = ax.imshow(images_list[j], cmap=cmap)
            else:
                img0 = ax.imshow(images_list[j], cmap=cmap,vmin=minval,vmax=maxval)
        if titles is not None:
            ax.set_title(titles[j])
        ax.axis("off")
        
        if show_colorbar:    
            divider = make_axes_locatable(ax)
            cax = divider.append_axes(cbarpos, size="5%", pad=0.05)
            plt.colorbar(img0, cax=cax)
            cax.yaxis.set_ticks_position(cbarpos)
    if scalebarval:
        scalebar = ScaleBar(scalebarval,
                            units=units,frameon=False,color='white',
                            location='lower right')
        ax.add_artist(scalebar)
    return fig,axes


def comparative_barplot(vals1,vals2,x,ax=None):
    """Compares two distributions side by side"""
    indices = np.arange(0,len(x))
    width = 0.35 
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.bar(indices,vals1,width,color='b')
    ax.bar(indices+width,vals2,width,color='r')
    
    ax.set_xticks(indices + width / 2)
    ax.set_xticklabels(x)
    return ax


def comparative_barplot_nd(values_list,x,ax=None):
    """Compares n distributions side by side"""
    indices = np.arange(0,len(x))
    width = 0.8/len(values_list) 
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    for j,val in enumerate(values_list):
        ax.bar(indices+width*j - width*len(values_list)/2,val,width)
    ax.set_xticks(indices)
    ax.set_xticklabels(x)
    return ax


def correction_comparison_plot(file,fig = None,cmap="hot",titles=None,repeat=True):
    out = comparison_file_extractor(file)
    filenames = out["filenames"]
    stacks = out["stacks"]
    sizex = out["psize_x"]
    sizez = out["psize_z"]
    
    def pn(stack_list,titles,fig):
        nt = len(titles)
        assert(len(stack_list)==nt)
        
        axes=[]
        
        for j in range(nt):
            axes.append(fig.add_subplot(1,nt,j+1))
        for j,ax in enumerate(axes):
            img0=ax.imshow(stack_list[j],cmap="hot")
            ax.set_title(titles[j])
            
            plt.axis("off")
            fig.colorbar(img0)
            #plt.colorbar()
            
    names =[ "".join(f.split(".")[:-1]) for f in filenames]
    names = [n.replace("\\","") for n in names]
    
    names = [n.replace("/","") for n in names]
    if fig is None:
        fig = plt.figure()
    if not repeat:
        to_eliminate = [j for j,name in enumerate(names[1:]) if name in names[:j-1]]
        for toe in to_eliminate:
            names.pop(toe)
    else:
        to_eliminate = []
        
    if titles is not None:
        names = titles
    
    to_exclude = [x for x,n in enumerate(names) if "confocal" in n ]
    stack_list = [stacks[i] for i in range(stacks.shape[0]) if i not in to_eliminate]
    
    stack_no_confocal = np.asarray(
            [st for name,st in zip(names,stack_list) if "confocal" not in name])
    stack_confocal = np.asarray(
            [st for name,st in zip(names,stack_list) if "confocal" in name])[0]
    
    mM = np.max(stack_no_confocal)
    mm = np.min(stack_no_confocal)
    
    f,a = colorbar_plot(stack_list,cmap=cmap,titles=names,maxval=mM,minval=mm,
                        scalebarval=sizez,fig=fig,scalefrees_list=to_exclude,nrows=2)
    #pn(stack_list,names,fig)
    nn =len(stack_list)
    
    ax=fig.add_subplot(2,nn,nn+1)
    image_overlay(stack_confocal,stacks[0],ax=ax)
    ax.set_title("Overlay 1")
    #+names[0]+" and confocal"
    ax=fig.add_subplot(2,nn,nn+2)
    ax.axis("off")
    image_overlay(stack_confocal,stacks[1],ax=ax)
    ax.set_title("Overlay 2")
    ax.axis("off")
    if sizez!=sizex:
        f.suptitle("WARNING: different x and z pixel size")

def aberration_difference_plot(file,fig = None,axes=None):
    """Used for comparison experiments"""
    if fig is None:
        fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    out = comparison_file_extractor(file)
    filenames = out["filenames"]
    names =[ "".join(f.split(".")[:-1]) for f in filenames]
    
    aberrations = out["aberrations"]
    
    aberrations = [ab for name,ab in zip(names,aberrations) if name!="confocal"]

    xab = np.arange(0,11)

    aberrations_list = [aberrations[i][xab] for i in range(len(aberrations))]
    
    axis = comparative_barplot_nd(aberrations_list,xab,ax=ax1)
    axis.legend(names)
    if len(aberrations)>=2:
        abdiff = aberrations[0]-aberrations[1] #In general we start with correction and then reference
        #modes = np.arange(abdiff.size)
        to_select=abdiff!=0
        modes = np.where(to_select)[0]
        indices = np.arange(0,np.count_nonzero(to_select))
        ax2.bar(indices,abdiff[to_select])
        ax2.set_xticks(indices)
        ax2.set_xticklabels(modes)
        
        
    ax1.set_title("Aberrations")
    ax2.set_title("Differences")
    ax1.set_xlabel("Mode")
    ax1.set_ylabel("Aberration (rms)")
    ax2.set_xlabel("Mode")  
    
def image_overlay(im1,im2,ax=None):
    if ax is None:
        fig,ax = plt.subplots()
        
    ax.imshow(im2/np.max(im2),cmap="Greens")
    ax.imshow(im1/np.max(im1),cmap = "hot",alpha=0.4)
    ax.axis("off")

def recenter_img(im,dx,dy):
    trf = tf.AffineTransform(translation = (dx,dy))
    v=tf.warp(im,trf)
    return v
def least_squares_registration(im1,im2):
    search_radius = 20 #pixels
    x1=np.arange(search_radius*2)-search_radius
    y1=np.arange(search_radius*2)-search_radius
    im1c = im1.copy()/np.max(im1)
    im2c = im2.copy()/np.max(im2)
    results = np.zeros((2*search_radius,2*search_radius))
    
    for i,x in enumerate(x1):
        for j,y in enumerate(y1):
            trf = tf.AffineTransform(translation = (x,y))
            tmp = tf.warp(im1c,trf)
            lq = np.sum((tmp-im2c)**2)
            results[i,j]=lq
    plt.figure()
    plt.imshow(results)
    results_map = results==np.min(results)
    assert(np.count_nonzero(results_map)==1)
    xo,yo = np.where(results_map)
    xo = x1[xo[0]]
    yo = y1[yo[0]]
    return xo,yo

def show_phase(aberration,plot=True):
    """Given an array of aberration coefficients, generates an image of the 
    corresponding phase.
    Parameters:
        aberration: numpy array, list of aberration coefficients in Noll order
        plot: bool, if True shows the phase mask in a separate plot
    Return:
        arr: 2D numpy array, the phase corresponding to the input aberration
        """
    slm = SLM()
    slm.set_aberration(aberration)
    arr = slm.phi
    arr[slm.rho>1.0]=np.nan
    if plot:
        plt.figure()
        plt.imshow(arr)
        plt.axis("off")
        plt.colorbar()
    return arr

def display_aberration_difference(ab):

    plt.figure()
    plt.subplot(122)
    arr=show_phase(ab,plot=False)
    plt.imshow(arr,)
    plt.axis("off")
    plt.title("Corrected pupil function")
    cbar = plt.colorbar()
    cbar.set_label("Phase (rad)")
    x = np.arange(ab.size)
    ab=ab.reshape(-1)
    x = x[ab!=0]
    ab = ab[ab!=0]
    plt.subplot(121)
    plt.bar(np.arange(x.size),ab,tick_label = x)
    plt.xlabel("Mode index")
    plt.ylabel("Value (rad)")

def bar_aberration_difference(ab,ax = None):
    """Bar plot of an aberration function ab. Shows only non zero values """
    if ax is None:
        fig,ax = plt.subplots(1,1)

    if ab is not None:
        print(ab.shape)
        print(ab)
        x = np.arange(ab.size)
        ab=ab.reshape(-1)
        x = x[~np.isclose(ab,0,atol=10**-3)]
        ab = ab[~np.isclose(ab,0,atol=10**-3)]
    else:
        x=[]
    if len(x)==0:
        x = np.arange(3,10)
        ab = np.zeros_like(x)
    ax.bar(np.arange(x.size),ab,tick_label = x)
    ax.set_xlabel("Mode index")
    ax.set_ylabel("Value (rad)")
    
def slx_errorbar(x,ys,names=[],normalise = False,start = 10**-5):
    """semilogx errorbar plot for FCS experiments.
    Parameters:
        x: numpy array, the time points (size N)
        ys: list, contains the datapoints in arrays with shape (p,N) with p the
        number of experiments and N the number of datapoints.
        names: list, the names of each plot
        normalise: bool, if True sets G(0) to 1 for each curve.
        start: if normalise is set to True, defines a minimum value for x
        """
        
    if len(names)<len(ys):
        for j in range(len(ys)-len(names)):
            names.append("")
    
    plt.figure()
    for j,y in enumerate(ys):
        if normalise:
            mean1 = np.mean(y,axis=0)
            mean1 = mean1[x>start]
            v0 = mean1[0]
            mean1 /= v0
            std1 = np.std(y/v0,axis=0)
            std1 = std1[x>start]
            xc = x[x>start]
            
        else:
            xc = x
            mean1 = np.mean(y,axis=0)
            std1 = np.std(y,axis=0)
        plt.errorbar(xc,mean1,yerr = std1*3, label = names[j])
    
    plt.xscale("log")
    plt.xlabel("τ (s)")
    plt.ylabel("G(τ)")
    plt.title("Adaptive optics for STED-FCS in KK")
    plt.legend()
    
def show_3Dprofiles(psf):
    plt.figure()
    plt.subplot(131)
    plt.imshow(psf[:,psf.shape[1]//2,:])
    plt.subplot(132)
    plt.imshow(psf[psf.shape[0]//2,:,:])
    plt.subplot(133)
    plt.imshow(psf[:,:,psf.shape[2]//2])
if __name__=="__main__":
  pass