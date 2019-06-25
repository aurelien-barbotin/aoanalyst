# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 12:01:38 2017

@author: Aurelien

This file contains functions that plot data from experiments for fast experiment
analysis.
"""
import matplotlib.pyplot as plt
import numpy as np
import math
import shutil
import os
import multipletau
import h5py

from itertools import chain
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit

from matplotlib_scalebar.scalebar import ScaleBar

from aoanalyst.io import file_extractor,comparison_file_extractor, general_file_extractor
from aoanalyst.misc import aberration_names,find_optimal_arrangement

from aoanalyst.sensorless import wavelet_metric,fitter_exp
from aoanalyst import mvect
from aoanalyst.graphics import interactive_plot,comparative_barplot,\
    correction_comparison_plot,image_overlay,display_aberration_difference



from matplotlib import colors as mcolors

def find_angles(im):
    """Finds the angle of the principal components in a grayscale image like an
    intensity profile using PCA.
    Parameters:
        im: numpy 2D array, """
    img = im.copy()
    shape = im.shape
    x = np.arange(shape[0])
    y = np.arange(shape[1])
    xx,yy=np.meshgrid(x,y)
    #xx=xx.reshape(-1)
    #yy=yy.reshape(-1)
    #To reduce number of points
    max_n_samples = 10**5
    while np.sum(img)>max_n_samples and np.max(img)>10:
        img = img//2
    n_samples = np.sum(img)
    X = np.zeros((n_samples,2))
    xind=0
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(img[i,j]):
                #print(im[i,j],i,j)
                X[xind,:]=np.array([i,j])
                xind+=1
    pc = PCA()
    pc.fit(X)
    return pc.components_,pc.mean_

def gaussian2D(x,xc,yc,sigma1,sigma2,a,c):
    """Generates a 2D gaussian modeling an illumination pattern, with possibly
    an asymetry.
    Parameters: 
        x: numpy 2D array. The distance values are stored in axis 0 and axis 1 
        differentiates between the two directions
        xc: float, center along the first direction
        yc: float, center along the second direction
        sigma1: float, gaussian standard deviation along the first direction
        sigma2: float, gaussian standard deviation along the second direction
        a: float, amplitude of the gaussian
        c: float, offset of the gaussian
    Returns:
        gaussian: numpy 1D array, the values of the gaussian evaluated in x"""
    xd1=x[:,0]-xc
    xd2=x[:,1]-yc
    return a*np.exp(-.5*( (xd1/sigma1)**2 + (xd2/sigma2)**2))+c
def fit_g2d(image):
    """Fits an intensity profile with an asymetric gaussian. Same as fit_intensity,
    except that it also returns the parameters of the fit
    Parameters:
        image: numpy 2D array, grayscale intensity profile
    Returns:
        fit: numpy 2D array, the image fitted
        popt: numpy 1D array, contains the parameters of the fit
        center: the center of the gaussian"""
    image_u8 = ((image/np.max(image))*255).astype(np.uint8)
    u,v = image.shape
    x = np.arange(u)
    y = np.arange(v)
    yy,xx=np.meshgrid(y,x)
    
    pc,mean=find_angles(image_u8)
    #To convert into rotation matrix
    if pc[0][0]*pc[1][1]<0:
        pc[0]*=-1
    #Generate the distances map for new basis
    x1= pc[0][0]*xx+pc[0][1]*yy
    x2 = pc[1][0]*xx+pc[1][1]*yy
    c1 = np.median(x1)
    c2 = np.median(x2)
    
    x1-=c1
    x2-=c2
    
    X = np.zeros((u*v,2))
    X[:,0] = x1.reshape(-1)
    X[:,1] = x2.reshape(-1)
    
    ydata = image.reshape(-1)
    
    popt,covs = curve_fit(gaussian2D,X,ydata)
    
    fit =  gaussian2D(X,*popt).reshape(u,v)
    Ri = np.array([[pc[1][1],-1*pc[0][1]],
                   [-1*pc[1][0],pc[0][0]]])
    
    center = np.array([popt[0],popt[1]])+np.array([c1,c2])
    center = np.dot(Ri,center)
    return fit,popt,center

# !!! Change
def sted_power(x):
    return x

def analyse_experiment(file,fig=None,interactive=None):
    """Plots the data from a modal AO experiment. 
    Parameters:
        file: string, filename
        interactive: int, optional. If not None, plots an interactive plot
        of the data for the corresponding mode. If interactive=4, the function
        will produce an interactive plot for the mode 4 (astigmatism)."""
    ext = file_extractor(file)

    try:
        xopt = ext['xopt']
    except:
        pass
        
    images = ext['log_images']
    modes = ext['modes']
    xdata = ext['xdata']
    ydata = ext['ydata']
    

    yhat = ext['yhat']
    corrected = ext['corrected_image']
    maxcorr = np.max(xdata)
    
    fitter_asym_bounds = (
        (-maxcorr, 0.0, 0.0, 0.0,0.0),
        (maxcorr, np.inf, np.inf, np.inf,np.inf))
    
    fitter_exp_bounds = (
        (-maxcorr, 0.0, 0.0, 0.0),
        (maxcorr, np.inf, np.inf, np.inf))
    
    Nz=len(modes)
    try:
        P = ext["P"]
    except:
        P = xdata.shape[0]
        
    if images.shape[0]%P!=0:
        corrected = images[-1,:,:]
    if type(corrected)==np.float64:
        corrected = images[-P//2]
    
    
    
    x_subplot = math.ceil(np.sqrt(Nz+2))
    if fig is None:
        fig = plt.figure()
    for i in range(Nz):
        ax = fig.add_subplot(x_subplot,x_subplot,i+1)
        images_sub = images[i*P:(i+1)*P]
    
        
        x=xdata[:,i]
        y = ydata[:,i]
        yh = yhat[:,i]
        """yw = wavelet_metric(images_sub)
        yw = yw/np.max(yw)"""
        
        ind_mvect=-1
        ymvect = []
        for j in range(P):
            vec = 1-mvect.compute_mvector(images_sub[j])[ind_mvect]
            ymvect.append(vec)
        ymvect = np.asarray(ymvect)/np.max(ymvect)
        
        try:
            popt, _ = curve_fit(
                fitter_exp, x, ymvect, p0=None, bounds=fitter_exp_bounds)
            yhh = fitter_exp(x, *popt)
        
        except:
            popt = 0*np.ones((len(fitter_exp_bounds[0]),))
            yhh = np.ones_like(x)
            print("fit failed",file,i)
            
        ax.plot(x,y,"bo")
        ax.plot(x,yh,color='red')
        ax.plot(x,ymvect,"go")
        ax.plot(x,yhh,color="green")
        #plt.plot(x,yw)
        ax.axvline(xopt[modes[i]],color='red')
        ax.axvline(popt[0],color="green")
        
        ax.set_title(aberration_names[modes[i]],y=1.08)
        
    fig.suptitle(file,y=1.08)
        
    ax.legend(["measured","fit","Mvect wavelet"])
    mM = max(np.max(images[P//2,:,:]),np.max(corrected))
    ax=fig.add_subplot(x_subplot,x_subplot,Nz+1)
    ax.imshow(images[P//2,:,:],cmap='inferno',vmax=mM)
    ax.set_title("Before correction",y=1.08)
    
    ax=fig.add_subplot(x_subplot,x_subplot,Nz+2)
    ax.imshow(corrected,cmap='inferno',vmax=mM)
    ax.set_title("After correction",y=1.08)
    
    if interactive is not None and interactive in modes:
        i = np.where(modes==interactive)[0][0]
        images_sub = images[i*P:(i+1)*P]
    
        
        x=xdata[:,i]
        y = ydata[:,i]
        yw = wavelet_metric(images_sub)
        ampl = np.max(yw)-np.min(yw)
        intensity = np.max(np.max(images_sub,axis=-1),axis=-1)
        intensity = intensity.astype(np.float)
        intensity = intensity-np.min(intensity)
        intensity = (intensity/np.max(intensity))*ampl+np.min(yw)
      
        ymvect = []
        for j in range(P):
            vec = 1-mvect.compute_mvector(images_sub[j])[ind_mvect]
            ymvect.append(vec)
        ymvect = np.asarray(ymvect)/np.max(ymvect)*np.max(yw)
        
        interactive_plot(x,yw,images_sub,supp_values=[intensity,ymvect])
        plt.title("mode"+str(interactive))
        plt.legend(["wavelets","intensity","mvector"])
        plt.show()
    return fig

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

def axvline2D(data,axis):
    try:
        for d in data:
            axis.axvline(d)
    except:
        axis.axvline(data)

def display_results(file,**kwargs):
    ext = general_file_extractor(file)
    if len(ext)==0:
        try:
            ext = comparison_file_extractor(file)
            if "psize_x" in ext:
                correction_comparison_plot(file,**kwargs)
            else:
                display_fcs_comparison_results(file,**kwargs)
            return
        except Exception as e:
            print("No matching display for you sir")
            print(e)
            return
    exp_class = ext["class"]
    if exp_class=="modal":
        if 'params' in ext.keys():
            f = display_experiment_results_cncb(file,**kwargs)
        else:
            f=display_experiment_results(file,show_ab_diff = False,**kwargs)
    elif exp_class=="fcs":
        f=display_fcs_results(file,**kwargs)
    elif exp_class=="correlation":
        if "log_z" in ext:
            f = display_z_correlation_results(file,**kwargs)
        else:
            f=display_xy_correlation_results(file,**kwargs)
    return f

def reorder(x,y):
    lists = sorted(zip(*[x, y]))
    new_x, new_y = list(zip(*lists))
    return new_x,new_y

def display_experiment_results(file,fig=None,supp_metrics=[],show_legend=True,
                               show_experimental=True,names=[],fitter=None,
                               show_ab_diff = True):
    """Plots the data from a modal AO experiment. 
    Parameters:
        file: string, filename
        fig: handle to the figure in which the data needs to be plotted. 
            if None, plots it in a new figure
        supp_metrics: list, optional. List of image quality metrics to be 
            evaluated on the dataset.
        show_legend: bool, optional. If True, displays the legend on the
            last subplot.
        show_experimental: bool, optional. If True, Displays the experimental
            (measured) data.
        names: list, optional. List of strings corresponding to the metrics
            names.
        fitter: Fitter, optional. Object of the class Fitter used to fit the 
            metric curves. If None, no fitting is applied.
    Returns: fig, handle to the figure created.
        """
    ext = file_extractor(file,open_stacks=False)
    if "log_autocorrelations" in ext:
        fast_fcs_results(file,fig=fig,show_legend=fig,
                               show_experimental=show_experimental,fitter=fitter,
                                                            autocorrelate = False)
        return fig
    else:
        
        ext = file_extractor(file)
    colors_ref = ["purple","black","green","orange","brown","pink","gray","olive",
              "cyan","blue","red"]
    class CircColor(object):
        def __init__(self):
            self.colors = colors_ref
        def __getitem__(self,nb):
            return self.colors[nb%len(self.colors)]
    colors = CircColor()
    
    images = ext['log_images']
    if "modes" in ext:
        modes = ext['modes']
    else:
        modes=[-1]
    if "P" in ext:
        P=ext["P"]
    if "xdata" in ext:
        xdata = ext['xdata']
        ydata = ext['ydata']
    else:
        xdata=np.arange(P*len(modes)).reshape(P,len(modes))
        ydata=np.ones((P,len(modes)))
        
    if "yhat" in ext:
        yhat = ext['yhat']
    else:
        yhat = ydata
    corrected_in=False #If True, there is a "corrected" image
    if "corrected_image" in ext:
        corrected = ext['corrected_image']
        corrected_in = True

    if "popt" in ext:
        popts = ext["popt"]
    else:
        popts=np.zeros((5,5))
    if "reference_aberration" in ext:
        reference_aberration = ext["reference_aberration"]
    else:
        reference_aberration = None
    xopt = None
    if "xopt" in ext:
        xopt = ext["xopt"]
    fitted_data = []
    legend = []
    
    Nz=len(modes)
    try:
        P = ext["P"]
    except:
        P = xdata.shape[0]
        
    if images.shape[0]%P==1:
        corrected = images[-1,:,:]
        if type(corrected)==np.float64:
            corrected = images[-P//2]
        
    #Sanity check
    if names is not None:
        if len(names)<=len(supp_metrics):
            for j in range(len(supp_metrics)-len(names)):
                names.append("_")
            
    sup_plots=1 #1 more plot for the histograms
    if corrected_in:
        sup_plots+=2
        
    x_subplot = math.ceil(np.sqrt(Nz+sup_plots))
    if fig is None:
        fig = plt.figure()
    #Prepare list containing axes
    axes=[]
    for i in range(Nz):
        axes.append(0)
    plots=[]
    
    
    for i in range(Nz):
        axes[i] = fig.add_subplot(x_subplot,x_subplot,i+1)
        ax = axes[i]
        x = xdata[:,i]
        y = ydata.transpose()[i].transpose()
        yh = yhat.transpose()[i].transpose()
        
        maxval=0
        
        
        if show_experimental:
            maxval = np.max(y)
            my = np.max(y,axis=0)
            y/=my
            y*=maxval
            p=ax.plot(reorder(x,y)[0],reorder(x,y)[1],marker="o")
            
            plots.append(p)
            mincorr = np.min(x)
            maxcorr=np.max(x)
            xhat = np.linspace(mincorr,maxcorr,yh.shape[0])
            p = ax.plot(xhat,maxval*yh,"--",color=colors[0])
            
            plots.append(p)
            if fitter is not None:
                poptf,xhf,yhf = fitter.fit(x,y)
                fitted_data.append((poptf[0],xhf,yhf))
                
        mode_images = images[i*P:(i+1)*P]
        
        ndpl = 0 #Number of n-dimensional plots added
        for l,met in enumerate(supp_metrics):
            
            out=[]
            for j in range(P):
                out.append(met(mode_images[j]))
            out = np.asarray(out)
            
            if maxval==0:
                maxval = np.max(out)
            out = out*maxval/np.max(out,axis=0)
            if out.ndim==1:
                line = ax.plot(reorder(x,out)[0],reorder(x,out)[1],
                               marker='^',color=colors[l+ndpl+1])
                plots.append(line)
            else:
                if i==0:
                    basename = names.pop(l+ndpl)
                    print(basename)
                for kk in range(out.shape[1]):
                    line = ax.plot(reorder(x,out[:,kk])[0],
                                   reorder(x,out[:,kk])[1],
                                   marker='^',color=colors[l+1+ndpl])
                    ndpl+=1
                    if i==0:
                        names.insert(l+ndpl+1,basename+str(ndpl))
                        print(names)
                    plots.append(line)
            
            if fitter is not None:
                poptf,xhf,yhf = fitter.fit(x,out)
                fitted_data.append((poptf[0],xhf,yhf))
        if i==0:
            legend = names[:]          
            
        if show_experimental:
            ax.axvline(popts[0,i],color='red')
        
        for j,(op,xhf,yhf) in enumerate(fitted_data):
            ax.plot(xhf,yhf*maxval/np.max(yhf,axis=0),'--') 
            ax.axvline(op)
            fitted_data = []
            
        ax.set_title(aberration_names[modes[i]],y=1.08)
    #legend = legend[:len(legend)//Nz]
    
    if show_experimental:
        tmp = ["measured","fit"]
        tmp.extend(legend)
        legend = tmp
        
    if show_legend:     
        #ax.legend(tuple(lines),tuple(legend),loc="lower right")
        fig.legend([x[0] for x in plots], tuple(legend[:len(plots)]), 'lower right')
        #ax.legend(legend)
    fig.suptitle(file,y=1.08)
    if corrected_in:
        if images.ndim!=3:
            for j in range(images.ndim-3):
                images=images[:,images.shape[1]//2]
                corrected=corrected[corrected.shape[0]//2]
                print(images.shape)
        mM = max(np.max(images[P//2,:,:]),np.max(corrected))
        ax2=fig.add_subplot(x_subplot,x_subplot,Nz+1)
        ax2.imshow(images[P//2,:,:],cmap='inferno',vmax=mM)
        ax2.set_title("Before correction",y=1.08)
        
        ax2=fig.add_subplot(x_subplot,x_subplot,Nz+2)
        ax2.imshow(corrected,cmap='inferno',vmax=mM)
        ax2.set_title("After correction",y=1.08)
    
    if reference_aberration is not None and show_ab_diff:
        firstmode=0
        lastmode=11
        if xopt is None:
            xopt = np.zeros_like(reference_aberration)
        xab = np.arange(firstmode,lastmode)
        yab = reference_aberration[firstmode:lastmode].reshape(-1)
        yab_corr = (reference_aberration+xopt)[firstmode:lastmode].reshape(-1)
        ax3=fig.add_subplot(x_subplot,x_subplot,Nz+sup_plots)
        #ax3.bar(xab,yab,color="blue")
        comparative_barplot(yab,yab_corr,xab,ax=ax3)
        ax3.set_title("Aberration values",y=1.08)
        ax3.legend(["Reference","Correction"])
    print("End display experiment result")
    return fig

def modal_summary(files):
    """Summarises a set of modal experiments.
    Parameters:
        files: list, each element is the name of one experiment file"""
    nplots=0
    for file in files:
        ext = file_extractor(file)
        modes = ext["modes"]
        nplots += len(modes)

    from aotools.ext.misc import find_optimal_arrangement
    n1,n2 = find_optimal_arrangement(nplots)
    fig,axes = plt.subplots(n1,n2)
    axes = axes.ravel()
    
    p = 0 #plot number

    for j in range(len(files)):
        file = files[j]
        ext = file_extractor(file)
        if p==0:
            reference_aberration = ext["reference_aberration"]
        modes = ext["modes"]
        xdata = ext["xdata"]
        ydata = ext["ydata"]
        yhat = ext["yhat"]
        popts = ext["popt"]
        
        plots=[]
        Nz = len(modes)
        for i in range(Nz):
            x = xdata[:,i]
            y = ydata[:,i]
            yh = yhat[:,i]        
            mincorr = np.min(x)
            maxcorr=np.max(x)
            xh = np.linspace(mincorr,maxcorr,yh.shape[0])
            a1 = axes[p].plot(x,y,"bo")
            a2 = axes[p].plot(xh,yh*np.max(y),"--")
            a3 = axes[p].axvline(popts[0,i],color='red')
            plots.append(a1[0])
            plots.append(a2[0])
            plots.append(a3)
            axes[p].set_title(str(p)+"/ "+aberration_names[modes[i]])
            axes[p].set_xlabel("Bias (rad)")
            axes[p].set_ylabel("Metric value")
            p+=1
            
    fig.legend([x for x in plots], tuple(["Measured","Fit","Correction"]), 'upper right')
    fig.tight_layout()
    correction = ext["reference_aberration"]+ext["xopt"]
    from aotools.ext.graphics import display_aberration_difference
    display_aberration_difference(correction-reference_aberration)
    
def analyse_pupilIntensiy(file,plot=False):
    """Analyses the data of an experiment measuring pupil intensity. It fits the
    measured intensity profile with a gaussian and returns its centre.
    Parameters:
        file: string, filename
        plot: bool, if True plots the results
    Returns:
        fit_centre: 2D numpy array, the coordinates of the centre of the beam in
        SLM coordinates"""
    stuff = file_extractor(file)
    images = stuff['log_images']
    max_displacement = stuff['max_displacement']
    pupil_xy = stuff["pupil_xy"]
    P = stuff["P"]
    #intensity = stuff['intensity']
    #scan_rho = stuff['scan_rho']

    u,v = images[0].shape
    m1= int(0.2*u)
    m2 = int(0.8*u)
    m3= int(0.2*v)
    m4 = int(0.8*v)
    images = images[:,m1:m2,m3:m4]
    
    max_intensities = np.asarray([np.percentile(x,90) for x in images])
    spe = int(np.sqrt(max_intensities.size))
    max_intensities = max_intensities.reshape(spe,spe)
    u,v = max_intensities.shape
    
    try:
        gfit,popt,fit_centre = fit_g2d(max_intensities)
    except:
        return (-1000,0)
    fit_centre = (2*max_displacement*fit_centre/P-max_displacement +pupil_xy)
    print("fit center:",fit_centre)
    
    x_p = np.linspace(-max_displacement, max_displacement, P) + pupil_xy[0] 
    y_p = np.linspace(-max_displacement, max_displacement, P) + pupil_xy[1] 
    
    xlab = x_p.astype(int)
    ylab = y_p.astype(int)
    
    ticks = np.linspace(0,P-1,3).astype(int)
    print(ticks,xlab)
    if plot:         
        fig = plt.figure()
        ax1 = fig.add_subplot(131)
        #intensities = gaussian_filter(intensities,1.5)
        imax1 = ax1.imshow(max_intensities )
        ax1.set_title("Max values")
        ax1.set_ylabel("x position")
        ax1.set_yticks(ticks )
        ax1.set_yticklabels(xlab[ticks])
        ax1.set_xlabel("y position")
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(ylab[ticks])
        cbar = plt.colorbar(imax1, extend='neither', spacing='proportional',
                orientation='horizontal', format="%.0f")
        
        ax2 = fig.add_subplot(132)
        imax2=ax2.imshow(gfit)
        ax2.set_title("Gaussian fit")
        ax2.set_ylabel("x position")
        ax2.set_yticks(ticks )
        ax2.set_yticklabels(xlab[ticks])
        ax2.set_xlabel("y position")
        ax2.set_xticks(ticks)
        ax2.set_xticklabels(ylab[ticks])
        cbar = plt.colorbar(imax2, extend='neither', spacing='proportional',
                orientation='horizontal', format="%.0f")
        
        ax3 = fig.add_subplot(133)
        imax3=ax3.imshow(np.abs(max_intensities-gfit))
        ax3.set_title("Absolute differences")
        ax3.set_ylabel("x position")
        ax3.set_yticks(ticks )
        ax3.set_yticklabels(xlab[ticks])
        ax3.set_xlabel("y position")
        ax3.set_xticks(ticks)
        ax3.set_xticklabels(ylab[ticks])
        cbar = plt.colorbar(imax3, extend='neither', spacing='proportional',
                orientation='horizontal', format="%.0f")
    
    def mosaic(images3D):
        n,u,v = images3D.shape
        nn = int(np.sqrt(n))
        out = np.zeros((u*nn,v*nn))
        assert(nn**2==n)
        
        for i in range(nn):
            for j in range(nn):
                out[i*u:(i+1)*u,j*v:(j+1)*v] = images3D[i*nn+j]
        return out
    
    if plot:
        ms = mosaic(images)
        plt.figure()
        plt.imshow(ms)
        plt.axis("off")
        plt.show()
    return fit_centre

def separate_experiments(files,folder1="ModalWFS",folder2="AberrationsInduced"):
    """Separates experiments between those that have a field "n_iterations", 
    which means that they are experiments in which we induced aberrations,
    and the others.
    Parameters:
        files: list, contains the names of the data files
        folder1: str, optional. The name of the folder in which the data from
        the modal experiments shall be copied.
        folder2: str,optional. The name of the folder in which the data from 
        the induced aberrations experiments shall be copied.
    """
    files.sort()
    file = files[0]
    filename = file.split("\\")[-1]
    filepath =  "".join(file.split("\\")[:-1])
    folder1 =filepath+"/"+folder1+"/"
    
    if not os.path.isdir(folder1):
        os.mkdir(folder1)
        
    folder2 = filepath+"/"+folder2+"/"
    if not os.path.isdir(folder2):
        os.mkdir(folder2)
    for i,file in enumerate(files):
        ext = file_extractor(file)
        filename = file.split("\\")[-1]
            
        if "n_iterations" in ext.keys():
            shutil.copyfile(file,folder2+str(i+1)+"_"+filename)
        else:
            shutil.copyfile(file,folder1+str(i+1)+"_"+filename)
        
def n_measurements_effect(file):
    """Studies how a change in the number of measurements can affect precision.
    Works for induced aberration experiments
    Parameters:
        file: str, filename"""
    ext = file_extractor(file)
    try:
        truth = ext["induced_aberration"]
    except:
        print("No induced aberration")
    corrections = ext["corrections"]
    images = ext['log_images']
    reference_aberration = ext["reference_aberration"]
    modes = ext['modes']
    
    corrected = ext['corrected_image']
    
    #truth = truth-reference_aberration
    P = ext['P']
    Nz = images.shape[0]//P
    xdata = ext['xdata']
    n_iterations = ext['n_iterations']
    maxcorr = np.max(xdata)
    
    min_p = 3
    xdatas=[]
    for j in range(min_p,P+1):
        xdatas.append(np.linspace(-maxcorr,maxcorr,j))
    def replaceWithCloser(val,arr):
        diff = np.abs(arr-val)
        return arr[diff==np.min(diff)][0]
    
    xdatas = [np.array(list(map(lambda x: replaceWithCloser(x,xdata[:,0]),xd))) for xd in xdatas]
    masks = [[xx in xd for xx in xdata[:,0]] for xd in xdatas]
    
    fitter_exp_bounds = (
        (-maxcorr, 0.0, 0.0, 0.0),
        (maxcorr, np.inf, np.inf, np.inf))
    fitter_asym_bounds = (
        (-maxcorr, 0.0, 0.0, 0.0,0.0),
        (maxcorr, np.inf, np.inf, np.inf,np.inf))
    
    opts=[]
    results={}
    results_j={}
    optjs=[]
    for x,mask in zip(xdatas,masks):
        opt_j=np.zeros(len(modes))
        for n in range(n_iterations):
            opt_j=np.zeros(len(modes))
            for m in range(len(modes)):
                i = m+n*len(modes)
                P = ext['P']
                images_sub = images[i*P:(i+1)*P][mask]
                P = np.sum(mask)
                ind_mvect=-1
                ymvect = []
                for j in range(P):
                    vec = 1-mvect.compute_mvector(images_sub[j])[ind_mvect]
                    ymvect.append(vec)
                ymvect = np.asarray(ymvect)/np.max(ymvect)
                yw = wavelet_metric(images_sub)
                yw = np.asarray(yw)
                yw /= np.max(yw)
                try:
                    popt2, _ = curve_fit(
                        fitter_exp, x, ymvect, p0=None, bounds=fitter_exp_bounds)
                    opt_j[m]+=popt2[0]
                except:
                    popt2 = 0*np.ones((len(fitter_exp_bounds[0]),))
                    #yhh = np.ones_like(xx)
                    print("fit jacopo failed",file,i)
                try:
                    popt, _ = curve_fit(
                        fitter_exp, x, yw, p0=None, bounds=fitter_exp_bounds)
        
                    opts.append(popt[0])                
                except:
                    popt = 0*np.ones((len(fitter_exp_bounds[0]),))
                    #yhh = np.ones_like(xx)
                    opts.append(np.nan)
                    print("fit aurel failed",file,i)
            optjs.append((x,n,opt_j.copy()))
    #Need to find where we are at each step:
    references = optjs[-n_iterations:]
    references = [x[-1] for x in references]
    references = np.cumsum(np.asarray(references),axis=0)[:n_iterations-1]
    references = np.concatenate(( np.zeros((1,references.shape[1]) ),references ),axis=0)

    optjs = [(u,v,w+references[v]) for (u,v,w) in optjs]
    return xdatas,masks,results,results_j,references,optjs

def analyse_experimental_set(correction_files,comparison_file,savenr=None):
    """Summarises a whole experiment, with the correction steps on the one hand
    and the before/after comparison on the other hand"""
    assert(type(correction_files==list))

    nmodes = 0  #Stores the number of modes corrected at all
    exts = []
    for file in correction_files:
        ext = file_extractor(file)
        exts.append(ext)
        nmodes+=len(ext["modes"])
    
    suppfigs=2    
    nmodes+=suppfigs
    
    m,n=find_optimal_arrangement(nmodes)
    index=0
    plt.figure()
    for j in range(len(exts)):
        ext = exts[j]
        xdata = ext["xdata"]
        ydata = ext["ydata"]
        yhat = ext["yhat"]
        modes = ext["modes"]
        popts = ext["popt"]
        Nz=len(modes)
        for i in range(Nz):
            index+=1
            plt.subplot(m,n,index)
            x=xdata[:,i]
            y = ydata.transpose()[i].transpose()
            yh = yhat.transpose()[i].transpose()
            
            y/=np.max(y,axis=0)
            plt.plot(x,y,marker="o")
            mincorr = np.min(x)
            maxcorr=np.max(x)
            xhat = np.linspace(mincorr,maxcorr,yh.shape[0])
            plt.plot(xhat,yh,"--")
            plt.title(str(index)+") "+aberration_names[modes[i]])
                        
            plt.axvline(popts[0,i],color='red')
    #Before/after in xy
    first_exp = exts[0]
    firstP = first_exp["P"]
    
    before_correction = first_exp["log_images"][firstP//2]
    after_correction = exts[-1]['corrected_image']
    try:
        assert(before_correction.shape==after_correction.shape)
        
        u,v = before_correction.shape
        composite = np.zeros((u,v*2))
        composite[:u,:v] = before_correction
        composite[:u,v:] = after_correction
        index+=1
        plt.subplot(m,n,index)
        plt.imshow(composite,cmap="magma")
        plt.axis("off")
        x = np.ones(u)*v
        y = np.arange(u)
        plt.plot(x,y,linestyle="--",color="white")
        plt.title("Before/After correction")
    except:
        pass
    
    #Show aberration difference
    index+=1
    ax2 = plt.subplot(m,n,index)
    out = comparison_file_extractor(comparison_file)
    filenames = out["filenames"]
    names =[ "".join(f.split(".")[:-1]) for f in filenames]
    
    aberrations = out["aberrations"]
    
    aberrations = [ab for name,ab in zip(names,aberrations) if name!="confocal"]
    if len(aberrations)>=2:
        abdiff = aberrations[0]-aberrations[1] #In general we start with correction and then reference
        #modes = np.arange(abdiff.size)
        to_select=abdiff!=0
        modes = np.where(to_select)[0]
        indices = np.arange(0,np.count_nonzero(to_select))
        ax2.bar(indices,abdiff[to_select])
        ax2.set_xticks(indices)
        ax2.set_xticklabels(modes)
        
        
    ax2.set_title("Total Correction")
    ax2.set_ylabel("Aberration (rms)")
    ax2.set_xlabel("Mode")  
    
    plt.tight_layout()
    
    correction_comparison_plot(comparison_file)
    plt.tight_layout()

def display_fcs_results(file,fig=None,supp_metrics=[],show_legend=True,
                               show_experimental=True,names=[],fitter=None,
                                                            autocorrelate = True):
    """Plots the data from a modal AO experiment. 
    Parameters:
        file: string, filename
        fig: handle to the figure in which the data needs to be plotted. 
            if None, plots it in a new figure
        supp_metrics: list, optional. List of image quality metrics to be 
            evaluated on the dataset.
        show_legend: bool, optional. If True, displays the legend on the
            last subplot.
        show_experimental: bool, optional. If True, Displays the experimental
            (measured) data.
        names: list, optional. List of strings corresponding to the metrics
            names.
        fitter: Fitter, optional. Object of the class Fitter used to fit the 
            metric curves. If None, no fitting is applied.
    Returns: fig, handle to the figure created.
        """
    colors = ["purple","black","green","orange","brown","pink","gray","olive",
              "cyan","blue","red"]
    ext = file_extractor(file)
    log_autocorrelations = None
    log_autocorrfits = None
    
    if "log_autocorrelations" in ext:
        log_autocorrelations = ext["log_autocorrelations"]
    if "log_autocorrfits" in ext:
        log_autocorrfits = ext["log_autocorrfits"]
    
        
    images = ext['log_images']
    
    if autocorrelate and \
    (log_autocorrelations is None or len(log_autocorrelations)!=images.shape[0]):
        log_autocorrelations = list()
        for st in images:
            autocorr = multipletau.autocorrelate(st,m=8,deltat=10**-3,normalize=True)
            log_autocorrelations.append(autocorr)
        log_autocorrelations=np.asarray(log_autocorrelations)
    if "modes" in ext:
        modes = ext['modes']
        
    else:
        modes=[-1]
        
    if "confocal" in ext:
        confocal = ext["confocal"]
    else:
        confocal = None
        
    if "P" in ext:
        P=ext["P"]
    if "xdata" in ext:
        xdata = ext['xdata']
        ydata = ext['ydata']
    else:
        xdata=np.arange(P*len(modes)).reshape(P,len(modes))
        ydata=np.ones((P,len(modes)))
        
    if "yhat" in ext:
        yhat = ext['yhat']
    else:
        yhat = ydata
    corrected_in=False #If True, there is a "corrected" image
    
    if "corrected_image" in ext:
        corrected = ext['corrected_image']
        corrected_in = True

    if "popt" in ext:
        popts = ext["popt"]
    else:
        popts=np.zeros((5,5))
    if "reference_aberration" in ext:
        reference_aberration = ext["reference_aberration"]
    else:
        reference_aberration = None
    xopt = None
    if "xopt" in ext:
        xopt = ext["xopt"]
    fitted_data = []
    legend = []
    
    Nz=len(modes)
    try:
        P = ext["P"]
    except:
        P = xdata.shape[0]
        
    if images.shape[0]%P==1:
        corrected = images[-1]
        if type(corrected)==np.float64:
            corrected = images[-P//2]
        
    #Sanity check
    if names is not None:
        if len(names)!=len(supp_metrics):
            names=None
            
    sup_plots=1 #1 more plot for the histograms

        
    x_subplot = Nz+sup_plots
    if fig is None:
        fig = plt.figure()
    #Prepare list containing axes
    axes=[]
    lines=[]
    
    for i in range(Nz):
        axes.append(0)
        
    for i in range(Nz):
        x=xdata[:,i]
        
        ax1 = fig.add_subplot(2,x_subplot,x_subplot+i+1)
        if log_autocorrelations is not None and len(log_autocorrelations)==len(images):
            mode_autocorr = log_autocorrelations[i*P:(i+1)*P]
            for j in range(P):
                ax1.semilogx(mode_autocorr[j,2:,0],mode_autocorr[j,2:,1],color=colors[j%len(colors)],\
                label = str(round(x[j],2) ))
                if log_autocorrfits is not None and len(log_autocorrfits)==len(log_autocorrelations):
                    autowash = log_autocorrfits[j+i*P]
                    fitval=G_corr(mode_autocorr[j,2:,0],*autowash)
                    ax1.semilogx(mode_autocorr[j,2:,0],fitval,color=colors[j%len(colors)],linestyle="--")
            ax1.legend()
        else:
            print("no autocorr")
        axes[i] = fig.add_subplot(2,x_subplot,i+1)
        ax=axes[i]
        y = ydata.transpose()[i].transpose()
        yh = yhat.transpose()[i].transpose()
        
        maxval=0
        
        
        if show_experimental:
            maxval = np.max(y)
            my = np.max(y,axis=0)
            y/=my
            y*=maxval
            p = ax.plot(x,y,marker="o",linestyle="")
            lines.append(p)
            mincorr = np.min(x)
            maxcorr=np.max(x)
            xhat = np.linspace(mincorr,maxcorr,yh.shape[0])
            p = ax.plot(xhat,maxval*yh,"--",color = colors[0])
            lines.append(p)
            if fitter is not None:
                poptf,xhf,yhf = fitter.fit(x,y)
                fitted_data.append((poptf[0],xhf,yhf))
                
        mode_images = images[i*P:(i+1)*P]
        for l,met in enumerate(supp_metrics):
            out=[]
            for j in range(P):
                out.append(met(mode_images[j]))
            out = np.asarray(out)
            if maxval==0:
                maxval = np.max(out)
            out = out*maxval/np.max(out)
            line=ax.plot(xdata,out,marker='^',color=colors[l+1],linestyle="")
            #print(line)
            lines.append(line)
            if fitter is not None:
                poptf,xhf,yhf = fitter.fit(x,out)
                fitted_data.append((poptf[0],xhf,yhf))
                
            #Filling the legend
            legend.append(names[l])
            if out.ndim==2:
                for m in range(out.shape[1]-1):
                    legend.append(names[l]+str(m+1))
                 
        if show_experimental:
            ax.axvline(popts[0,i],color='red')
        
        for j,(op,xhf,yhf) in enumerate(fitted_data):
            print("Fitted data")
            ax.plot(xhf,yhf*maxval/np.max(yhf,axis=0),'--') 
            ax.axvline(op)
            fitted_data = []
            
        ax.set_title(aberration_names[modes[i]],y=1.08)
   
    legend = legend[:len(legend)//Nz]
    
    if show_experimental:
        tmp = ["measured","fit"]
        tmp.extend(legend)
        legend = tmp
        
    if show_legend:     
        #ax.legend(tuple(lines),tuple(legend),loc="lower right")
        
        fig.legend([x[0] for x in lines], tuple(legend), 'lower right')
    if confocal is not None:
        ax2=fig.add_subplot(2,x_subplot,x_subplot)
        try:
            autocorr_confocal = multipletau.autocorrelate(confocal,\
                                                          deltat=10**-3,\
                                                          m=8,\
                                                          normalize=True)
            if corrected_in:
                autocorr_corrected = log_autocorrelations[-2]
            else:
                autocorr_corrected= log_autocorrelations[-P//2]
            autocorr_reference = log_autocorrelations[P//2]
            xx=mode_autocorr[0,1:,0]
            ax2.semilogx(xx,autocorr_confocal[1:,1])
            if corrected_in:
                ax2.semilogx(xx,autocorr_corrected[1:,1])
            ax2.semilogx(xx,autocorr_reference[1:,1])
            
            print("Snooop dogg")
            ax2.legend(["Confocal","After correction","Before correction"])
        except Exception as e:
            print("Snooop doggy dogg")
            print(e)
        
    if reference_aberration is not None:
        firstmode=0
        lastmode=11
        if xopt is None:
            xopt = np.zeros_like(reference_aberration)
        xab = np.arange(firstmode,lastmode)
        yab = reference_aberration[firstmode:lastmode].reshape(-1)

        yab_corr = (reference_aberration+xopt)[firstmode:lastmode]
        yab_corr = yab_corr.reshape(-1)
        ax3=fig.add_subplot(2,x_subplot,2*x_subplot)
        #ax3.bar(xab,yab,color="blue")
        comparative_barplot(yab,yab_corr,xab,ax=ax3)
        ax3.set_title("Aberration values",y=1.08)
        ax3.legend(["Reference","Correction"])
        
    
    return fig


def display_fcs_comparison_results(file, fig=None, plot_comp = False, show_legend = True,
                                   **kwargs):
    ext = comparison_file_extractor(file)
    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(111)
    stacks = ext["stacks"]
    
    npl = stacks.shape[0]
    
    fig.suptitle(ext["notes"])
    filenames = ext["filenames"] 
    
    dt=10**-6
    
    names = [os.path.split(x)[-1].split(".")[0] for x in filenames]
    
    for j in range(npl):
        dat = stacks[j]
        corr = multipletau.autocorrelate(dat,deltat=dt,normalize=True)
        ax.semilogx(corr[2:,0], corr[2:,1],label=names[j])
        ax.set_xlabel("τ")
        ax.set_ylabel("G(τ)")
    #brightness = lambda x: np.mean(x)/molecular_number_autocorr(x)
    ax.legend()
    if plot_comp:
        diff = ext["aberrations"][1]-ext["aberrations"][0]
        display_aberration_difference(diff)
    return fig

def display_xy_correlation_results(file,fig=None,**kwargs):
    ext = general_file_extractor(file)
    
    log_confocal = ext["log_confocal"]

    if "log_2ds" in ext.keys() and ext["log_2ds"].size!=0:
        logs = ext["log_2ds"]
    else:
        logs=ext["log_3ds"]
    log_shifts = ext["log_shifts"]
    log_comas = ext["log_tiptilt"]
    psize = ext["psize"]
    
    shiftsx = [x[0] for x in log_shifts]
    shiftsy = [x[1] for x in log_shifts]
    
    tip = [x[0] for x in log_comas]
    tilt = [x[1] for x in log_comas]
    if fig is None:
        plt.figure()
        plt.subplot(121)
        plt.plot(tip,shiftsx,"-bo")
        plt.title("Tip")
        plt.xlabel("Bias (rad)")
        plt.ylabel("Shift (pixels)")
        plt.subplot(122)
        plt.plot(tilt,shiftsy,"-bo")
        plt.title("Tilt")
        plt.xlabel("Bias (rad)")
        plt.ylabel("Shift (pixels)")
    
    nbias = logs.shape[0]
    if fig is None:
        fig, axes =plt.subplots(nrows=1,ncols=nbias)
    else:
        axes=[]
        for i in range(nbias):
            ax=fig.add_subplot(1,nbias,i+1)
            axes.append(ax)
    confocal = log_confocal[0]
    for j in range(nbias):
        comval = str(round(log_comas[j][0],2) )+", "+ str( round(log_comas[j][1],2) )
        stedim = logs[j]
        ax0 = axes[j]
        
        image_overlay(confocal,stedim,ax=ax0)
        ax0.set_title("tip, tilt: "+comval)
    scalebar = ScaleBar(psize,
                units='nm',frameon=False,color='white',
                location='lower right')
    ax0.add_artist(scalebar)
    return fig

def display_z_correlation_results(file,fig=None,**kwargs):
    ext = general_file_extractor(file)
    psize = ext["psize"]
    log_confocal = ext["log_confocal"]

    if "log_2ds" in ext.keys() and ext["log_2ds"].size!=0:
        logs = ext["log_2ds"]
    else:
        logs=ext["log_3ds"]
    log_bias = ext["log_z"]
    logs = ext["log_3ds"]

    
    nbias = logs.shape[0]
    print(nbias)
    if fig is None:
        fig, axes =plt.subplots(nrows=1,ncols=nbias)
    else:
        axes=[]
        for i in range(nbias):
            ax=fig.add_subplot(1,nbias,i+1)
            axes.append(ax)
    confocal = log_confocal[0]
    defoc_nm = 339
    for j in range(nbias):
        comval = str(round(defoc_nm*log_bias[j][0],2))
        stedim = logs[j]
        ax0 = axes[j]
        
        image_overlay(confocal,stedim,ax=ax0)
        ax0.set_title("Defocus "+comval+" nm")
    scalebar = ScaleBar(psize,
                    units='nm',frameon=False,color='white',
                    location='lower right')
    ax0.add_artist(scalebar)
    return fig

def display_fcs_calibration(file, fig = None, autocorrelate = True,
                            dt = None, start = 2*10**-6,**kwargs):
    if dt is None:
        dt = 10**-6
    
    
    ext = general_file_extractor(file)
    
    confocal = ext["confocal"]
    
    powers = ext["powers"]
    notes=ext["notes"]
    stacks = ext["stacks"]
    
    intensities = np.sum(stacks,axis=tuple(np.arange(1,stacks.ndim)))
    
    powers_mw = sted_power(powers)
    
    plt.figure()
    plt.plot(powers_mw,intensities)
    plt.xlabel("STED power (mW)")
    plt.ylabel("Intensity")
    
    
    powers_mw = np.concatenate((np.zeros(1),powers_mw))
    stacks = np.concatenate((confocal.reshape(1,-1),stacks),axis=0)
    
    if autocorrelate:
        plt.figure()
        for j in range(stacks.shape[0]):
            st = stacks[j]
            if st.ndim>1:
                print("Alert: shape of data is",st.shape)
                st = st.reshape(-1)
            corr = multipletau.autocorrelate(st,deltat=dt,normalize=True)
            x = corr[1:,0]
            y = corr[1:,1]
            
            y = y[x>start]
            x = x[x>start]
            plt.semilogx(x,y)
        plt.legend([str(v) for v in powers_mw])
        plt.xlabel("τ (s)")
        plt.ylabel("G(τ)")
        plt.title(notes)
        
def display_STED_FCS_comparison(file,nsplit=6,first=50000,savefig = True,fit=True,
                                tauT = 5*10**-3, tauxy = 0.21):
    """Quickly displays the result of an experiment where different correction files
    are used at various STED powers"""
    ext = comparison_file_extractor(file)
    print(ext.keys())
    dt = 10**-3
    
    linestyles=[":","-.","--"]
    linecount=0

    colors = list(mcolors.BASE_COLORS)

    confocal = ext["confocal"]
    if confocal.ndim==1:
        confocal = confocal.reshape(1,-1)
    powers = ext["powers"]
    notes=ext["notes"]
    stacks = ext["stacks"][:,:,first:]
    filenames =ext["filenames"]
    
    names = [os.path.split(x)[-1].split(".")[0] for x in filenames]
    powers_mw = sted_power(powers)

    corrsconf=[]
    for conf in confocal:
        conf = conf[first:]
        conf_spl= np.split(conf[:conf.size-conf.size%nsplit],nsplit)
        for confsplit in conf_spl:
            corrconf = multipletau.autocorrelate(confsplit,deltat=dt,normalize=True)
            corrsconf.append(corrconf)
    corrconf1 = np.asarray(corrsconf[:len(corrsconf)//2])
    corrconf1 = np.median(corrconf1,axis=0)
    corrconf2 = np.asarray(corrsconf[len(corrsconf)//2:])
    corrconf2 = np.median(corrconf2,axis=0)

    all_fit_parameters = []
    plt.figure()
    for k in range(stacks.shape[1]): #Iterate over powers
        for j in range(stacks.shape[0]):
            st = stacks[j][k]
            st_split = np.split(st[:st.size-st.size%nsplit],nsplit)
            corrs = []
            pars = []
            for splitted_stack in st_split:
                print(j,k)
                corr = multipletau.autocorrelate(splitted_stack,m=8,deltat=dt,normalize=True)
                parfits, G = lmfit_z(corr[2:,0],corr[2:,1],triplet_time = tauT, txy = tauxy)
                pars.append(parfits)
                corrs.append(corr)
                
            pars = np.asarray(pars)
            all_fit_parameters.append((j,k,pars))
            corrs = np.asarray(corrs)
            corr = np.median(corrs,axis=0)
            if st.ndim>1:
                print("Alert: shape of data is",st.shape)
                st = st.reshape(-1)
            x = corr[2:,0]
            y = corr[2:,1]
            if names[j]==names[0]:
                ls = "-"
                linecount=0
            else:
                ls = linestyles[linecount]
                linecount += 1
            meanfit = np.mean(np.array([ G(x,*pa) for pa in pars]),axis=0)
            if k==0:
                plt.subplot(121)
                plt.semilogx(x,y,
                             label = names[j] +" "+ str(round(powers_mw[k],2)) +"mW",
                             color = colors[k],linestyle =ls )
                plt.semilogx(x,meanfit,linestyle = "--",color="gray")
                plt.subplot(122)
                plt.semilogx(x,y/np.mean(y[:5]),label = names[j] +" "+ str(round(powers_mw[k],2)) +"mW",color = colors[k],linestyle =ls )
            else: #No label
                plt.subplot(121)
                plt.semilogx(x,y,color = colors[k],linestyle =ls )
                plt.semilogx(x,meanfit,linestyle = "--",color="gray")
                plt.subplot(122)
                if j==0:
                    plt.semilogx(x,y/np.mean(y[:5]),
                                 label = str(round(powers_mw[k],2)) +"mW",
                                 color = colors[k],linestyle =ls )
                else:
                    plt.semilogx(x,y/np.mean(y[:5]),
                                 color = colors[k],linestyle =ls )
        plt.xlabel("τ")
        plt.ylabel("G(τ)")
    try:
        x = corrconf1[2:,0]
        y = corrconf1[2:,1]
        plt.subplot(121)
        plt.semilogx(x,y,label = "confocal before",linestyle = "-.",marker="x" )
        plt.subplot(122)
        plt.semilogx(x,y/np.mean(y[0:5]),label = "confocal before",linestyle = "-.",marker="x" )
    
    
        x = corrconf2[2:,0]
        y = corrconf2[2:,1]
        plt.subplot(121)
        plt.semilogx(x,y,label = "confocal after",linestyle = "-." ,marker="x")
        plt.subplot(122)
        plt.semilogx(x,y/np.mean(y[0:5]),label = "confocal after",linestyle = "-.",marker="x" )
 
    except:
        print("no conf after")
    plt.legend()
    plt.suptitle(notes)
    if savefig:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

        fname = os.path.split(file)[-1].split(".")[0]
        plt.savefig(fname+"_diff_powers")
    
    #Populate dict with results
    fnames = set(filenames)
    data = [ [[] for p in powers] for w in fnames]
    fit_results = dict(zip(fnames,data))
    for j,k,pars in all_fit_parameters:
        fn = filenames[j]
        fit_results[fn][k].append(pars)
        
    plt.figure()
    for fnm in fit_results.keys():
        dataPow = fit_results[fnm]
        name = os.path.split(fnm)[-1].split(".")[0]
        dataPow = [np.concatenate(w,axis=0) for w in dataPow]

        plt.subplot(121)
        plt.errorbar(sted_power(powers), np.mean(dataPow,axis=1)[:,0],yerr = 
                     np.std(dataPow,axis=1)[:,0], label=name, capsize = 5)
        plt.xlabel("STED power(mW)")
        plt.title("N")
        plt.subplot(122)
        plt.errorbar(sted_power(powers), np.mean(dataPow,axis=1)[:,1],yerr = 
                     np.std(dataPow,axis=1)[:,1], label=name, capsize = 5)
        plt.xlabel("STED power(mW)")
        plt.title("AR")
    plt.legend()
    if savefig:
        fname = os.path.split(file)[-1].split(".")[0]
        plt.savefig(fname+"_fits")
        
def display_STED_FCS_comparison_fitter(file,fitter,first=50000,savefig=True):
    """Quickly displays the result of an experiment where different correction files
    are used at various STED powers"""
    ext = comparison_file_extractor(file)
    print(ext.keys())
    dt = 10**-3
    
    linestyles=[":","-.","--"]
    linecount=0

    colors = list(mcolors.BASE_COLORS)

    confocal = ext["confocal"]
    if confocal.ndim==1:
        confocal = confocal.reshape(1,-1)
    powers = ext["powers"]
    notes=ext["notes"]
    stacks = ext["stacks"][:,:,first:]
    filenames =ext["filenames"]
    
    names = [os.path.split(x)[-1].split(".")[0] for x in filenames]
    powers_mw = sted_power(powers)

    corrsconf=[]
    for conf in confocal:
        conf = conf[first:]
        corrconf = multipletau.autocorrelate(conf,deltat=dt,normalize=True)


    all_fit_parameters = []
    plt.figure()
    for k in range(stacks.shape[1]): #Iterate over powers
        for j in range(stacks.shape[0]):
            st = stacks[j][k]
            corrs = []
            pars = []
            corr = multipletau.autocorrelate(st,m=8,deltat=dt,normalize=True)
            parfits, G = fitter(corr[2:,0],corr[2:,1])
            pars.append(parfits)
            corrs.append(corr)
                
            pars = np.asarray(pars)
            all_fit_parameters.append((j,k,pars))
            corrs = np.asarray(corrs)
            corr = np.median(corrs,axis=0)
            if st.ndim>1:
                print("Alert: shape of data is",st.shape)
                st = st.reshape(-1)
            x = corr[2:,0]
            y = corr[2:,1]
            if names[j]==names[0]:
                ls = "-"
                linecount=0
            else:
                ls = linestyles[linecount]
                linecount += 1
            meanfit = np.mean(np.array([ G(x,*pa) for pa in pars]),axis=0)
            if k==0:
                plt.subplot(121)
                plt.semilogx(x,y,
                             label = names[j] +" "+ str(round(powers_mw[k],2)) +"mW",
                             color = colors[k],linestyle =ls )
                plt.semilogx(x,meanfit,linestyle = "--",color="gray")
                plt.subplot(122)
                plt.semilogx(x,y/np.mean(y[:5]),label = names[j] +" "+ str(round(powers_mw[k],2)) +"mW",color = colors[k],linestyle =ls )
            else: #No label
                plt.subplot(121)
                plt.semilogx(x,y,color = colors[k],linestyle =ls )
                plt.semilogx(x,meanfit,linestyle = "--",color="gray")
                plt.subplot(122)
                if j==0:
                    plt.semilogx(x,y/np.mean(y[:5]),
                                 label = str(round(powers_mw[k],2)) +"mW",
                                 color = colors[k],linestyle =ls )
                else:
                    plt.semilogx(x,y/np.mean(y[:5]),
                                 color = colors[k],linestyle =ls )
        plt.xlabel("τ")
        plt.ylabel("G(τ)")
    try:
        x = corrconf[2:,0]
        y = corrconf[2:,1]
        plt.subplot(121)
        plt.semilogx(x,y,label = "confocal",linestyle = "-.",marker="x" )
        plt.subplot(122)
        plt.semilogx(x,y/np.mean(y[0:5]),label = "confocal",linestyle = "-.",marker="x" )
    
    
 
    except:
        print("erroconf")
    plt.legend()
    plt.suptitle(notes)
    if savefig:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

        fname = os.path.split(file)[-1].split(".")[0]
        plt.savefig(fname+"_diff_powers")
    
    #Populate dict with results
    fnames = set(filenames)
    data = [ [[] for p in powers] for w in fnames]
    fit_results = dict(zip(fnames,data))
    for j,k,pars in all_fit_parameters:
        fn = filenames[j]
        fit_results[fn][k].append(pars)
    
    npars=len(parfits)-2
    plt.figure()
    for fnm in fit_results.keys():
        dataPow = fit_results[fnm]
        name = os.path.split(fnm)[-1].split(".")[0]
        dataPow = [np.concatenate(w,axis=0) for w in dataPow]
        for k in range(npars):
            plt.subplot(1,npars,k+1)
            plt.errorbar(sted_power(powers), np.mean(dataPow,axis=1)[:,k],yerr = 
                         np.std(dataPow,axis=1)[:,k], label=name, capsize = 5)
        plt.xlabel("STED power(mW)")
        plt.title("parameter"+str(k))
    plt.legend()
    if savefig:
        fname = os.path.split(file)[-1].split(".")[0]
        plt.savefig(fname+"_fits")
def test_STED_FCS_comparison(file,savefig=False):
    """Verifies wether the STED-FCS experiments are repeatable: do we have the 
    same brightness and stds for identical measurements at different times"""
    ext = comparison_file_extractor(file)
    figname = os.path.split(file)[-1].split(".")[0]
    print(ext.keys())

    def spst(array,n):
        return np.split(array[:array.size-array.size%n],n)

    confocal = ext["confocal"]
    if confocal.ndim==1:
        confocal = confocal.reshape(1,-1)
    confocal = confocal[:,50000:]
    powers = ext["powers"]
    notes=ext["notes"]
    stacks = ext["stacks"][:,:,50000:]
    #Shape: (file,power,data)
    filenames =ext["filenames"]
    names = [os.path.split(x)[-1].split(".")[0] for x in filenames]
    
    results = set(filenames)
    results = dict(zip(results,[[] for x in results]))
    
    
    conf_means = list(chain.from_iterable( map(np.mean,spst(x,10))  for x in confocal))
    #conf_stds = list(chain.from_iterable( map(np.std,spst(x,10))  for x in confocal))
    plt.figure()
    plt.plot(conf_means)
    plt.suptitle("Confocal shape:"+str(confocal.shape))
    if savefig:
        plt.savefig(figname+"Confocal")
    plt.figure()
    for k in range(stacks.shape[1]): # iteration over powers
        stk = stacks[:,k,:]
        means_file1 = list(chain.from_iterable( map(np.mean,spst(x,10))  for x in stk[::2,:]))
        means_file2 = list(chain.from_iterable( map(np.mean,spst(x,10))  for x in stk[1::2,:]))
    
        means_file1 = np.asarray(means_file1)/means_file1[0]
        means_file2 = np.asarray(means_file2)/means_file2[0]
        std_file1 = list(chain.from_iterable( map(np.std,spst(x,10))  for x in stk[::2,:]))
        std_file2 = list(chain.from_iterable( map(np.std,spst(x,10))  for x in stk[1::2,:]))
        std_file1 = np.asarray(std_file1)/std_file1[0]
        std_file2 = np.asarray(std_file2)/std_file2[0]
        
        plt.subplot(121)
        plt.plot(means_file1,"o")
        plt.title("Mean "+names[0])
        plt.axvline(means_file1.size/2-0.5)

        plt.subplot(122)
        plt.plot(means_file2,"o",label=str(powers[k])+" mV")
        plt.title("Mean "+names[1])
        plt.axvline(means_file1.size/2-0.5)
        plt.xlabel("measurement nr")
        plt.ylabel("value (normalised to 1st value)")
    plt.suptitle(notes)
    plt.legend()
    if savefig:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.savefig(figname+"_STED")
        
        
def powers_calibrations_comparison(file, nsplit=6, savefig=False, verbose = False):
        """Quickly displays the result of an experiment where different correction files
        are used at various STED powers. Shows the results in different windows.
        Parameters:
            file: str
            nsplit: int, each timetrace is splitted nsplit times
            savefig: bool, if True saves the resulting curves in a separate file
            verbose: bool
        """
        ext = comparison_file_extractor(file)
        print(ext.keys())
        dt = 10**-3
        
        colors = list(mcolors.BASE_COLORS)
    
        confocal = ext["confocal"]
        
        powers = ext["powers"]
        notes=ext["notes"]
        stacks = ext["stacks"]
        filenames =ext["filenames"]
        
        names = [os.path.split(x)[-1].split(".")[0] for x in filenames]
        powers_mw = sted_power(powers)
        
        #powers_mw = np.concatenate((np.zeros(1),powers_mw))
        
        correlations=[]
        
        fig1,axes1 = plt.subplots(1,2)
        if savefig:
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
        fig2,axes2 = plt.subplots(1,2)
        if savefig:
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
        
        for k in range(stacks.shape[1]): #Iterate over powers
            for j in range(stacks.shape[0]):
                st = stacks[j][k]
                st_split = np.split(st[:st.size-st.size%nsplit],nsplit)
                corrs = []
                for splitted_stack in st_split:
                    if verbose:
                        print(j,k)
                    corr = multipletau.autocorrelate(splitted_stack,m=16,deltat=dt,normalize=True)
                    corrs.append(corr)
                corrs = np.asarray(corrs)
                corr = np.median(corrs,axis=0)
                if st.ndim>1:
                    print("Alert: shape of data is",st.shape)
                    st = st.reshape(-1)
                x = corr[1:,0]
                y = corr[1:,1]
                correlations.append((k,j,y))
                
                ls = "-"
                if names[j]==names[0]:
                    axes = axes1
                else:
                    axes = axes2
                #plt.subplot(121)
                if j in [0,1]:
                    axes[0].semilogx(x,y,label = names[j] +" "+ str(round(powers_mw[k],2)) +"mW",color = colors[k],linestyle =ls )
                    #plt.subplot(122)
                    axes[1].semilogx(x,y/np.mean(y[:5]),label = names[j] +" "+ str(round(powers_mw[k],2)) +"mW",color = colors[k],linestyle =ls )
                else:
                    axes[0].semilogx(x,y,color = colors[k],linestyle =ls )
                    #plt.subplot(122)
                    axes[1].semilogx(x,y/np.mean(y[:5]),color = colors[k],linestyle =ls )
            axes1[0].set_xlabel("τ")
            axes1[0].set_ylabel("G(τ)")
            axes1[1].set_xlabel("τ")
            axes1[1].set_ylabel("G(τ)")
            
            axes2[0].set_xlabel("τ")
            axes2[0].set_ylabel("G(τ)")
            axes2[1].set_xlabel("τ")
            axes2[1].set_ylabel("G(τ)")
        corrsconf=[]
        for conf in confocal:
            conf = conf[50000:]
            conf_spl= np.split(conf[:conf.size-conf.size%nsplit],nsplit)
            for confsplit in conf_spl:
                corrconf = multipletau.autocorrelate(confsplit,deltat=dt,normalize=True)
                corrsconf.append(corrconf)
        
        corrconf1 = np.asarray(corrsconf[:len(corrsconf)//2])
        corrconf1 = np.median(corrconf1,axis=0)
        corrconf2 = np.asarray(corrsconf[len(corrsconf)//2:])
        corrconf2 = np.median(corrconf2,axis=0)
        x = corrconf1[1:,0]
        y = corrconf1[1:,1]
        #plt.subplot(121)
        axes1[0].semilogx(x,y,label = "confocal before",linestyle = "-." )
        axes2[0].semilogx(x,y,label = "confocal before",linestyle = "-." )
        #plt.subplot(122)
        axes1[1].semilogx(x,y/np.mean(y[0:5]),label = "confocal before",linestyle = "-." )
        axes2[1].semilogx(x,y/np.mean(y[0:5]),label = "confocal before",linestyle = "-." )
        x = corrconf2[1:,0]
        y = corrconf2[1:,1]
        #plt.subplot(121)
        axes1[0].semilogx(x,y,label = "confocal after",linestyle = "-." )
        axes2[0].semilogx(x,y,label = "confocal after",linestyle = "-." )
        axes1[0].set_title("Raw")
        axes2[0].set_title("Raw")
        #plt.subplot(122)
        axes2[1].semilogx(x,y/np.mean(y[0:5]),label = "confocal after",linestyle = "-." )
        axes1[1].semilogx(x,y/np.mean(y[0:5]),label = "confocal after",linestyle = "-." )
        axes1[1].set_title("Normalised")
        axes2[1].set_title("Normalised")
    
        axes1[1].legend()
        axes2[1].legend()
        fig1.suptitle(names[0])
        fig2.suptitle(names[1])

        if savefig:
            plt.show()
            fname = os.path.split(file)[-1].split(".")[0]
            fig1.savefig(fname+"_"+names[0])
            fig2.savefig(fname+"_"+names[1])
            
        return x,correlations,[corrconf1,corrconf2],notes
    
def aberration_different_depths(depths,files,ref="4um",idf_mat = None,
                                axes=None,colors=None):
    """Plots the evolution of aberration vs depth in multi depths experiments.
    Parameters:
        depths, list, contains the different depths probed
        files: list, contains the path to the experiment files
        ref: optional, default = "4um". str, key to find in correction file defining
        the reference
        idf_mat: numpy array, inverse transofrm matrix
        axes: optional, default = None.list of axes where the results will be 
            plotted. if None, creates a new figure.
        colors: optional, default = None. colors for each depth"""
    all_aberrations = []
    zshift = 339
    xyshift = 172
    for dz,f in zip(depths,files):
        print("Depth",dz)
        
        ext = comparison_file_extractor(f,open_stacks=False)
        aberrations = ext["aberrations"]
        print(ext["filenames"])
        print("Stacks in ext:","stacks" in ext)
        for j,cfile in enumerate(ext["filenames"]):
            print(cfile)
            if ref in cfile:
                ab1 = aberrations[j]
            elif str(dz)+"um" in cfile:
                ab2 = aberrations[j]
        all_aberrations.append((dz,ab1,ab2))
    
    #plotting
    if axes is None:
        f, (ax0, ax1) = plt.subplots(1,2, gridspec_kw = {'width_ratios':[1, 2]})
    else:
        ax0, ax1 = axes
    N = len(all_aberrations)
    width = 0.9/N
    #abmodes = [1,2,4,5,6,7,10,21,36]
    
    abmodes0 = [1,2,0]
    abmodes1 = [4,5,6,7,10,21]
    shifts = np.array([xyshift,xyshift,zshift])
    nm1 = len(abmodes0)
    nm2 = len(abmodes1)
    for j,(dz,ref,cor) in enumerate(all_aberrations):
        print(dz)
        positions0 = np.arange(nm1)
        positions1 = np.arange(nm2)
        col = None
        if colors is not None:
            col = colors[j]
                
        if idf_mat is None:
            ax0.bar(positions0 + j*width, (ref-cor)[abmodes0]*shifts, width,
                    label=str(dz)+" µm", color=col)
            ax1.bar(positions1 + j*width, (ref-cor)[abmodes1], width,
                    label=str(dz)+" µm", color=col)
        else:
            print("idfmat:")
            print("0,1,2 before:",(ref-cor)[[0,1,2]])
            ab = np.dot(idf_mat,(ref-cor))
            print("0,1,2 after:",ab[[0,1,2]])

            ax0.bar(positions0 + j*width, ab[abmodes0]*shifts, width,
                    label=str(dz)+" µm",color=col)
            ax1.bar(positions1 + j*width, ab[abmodes1], width,
                    label=str(dz)+" µm", color=col)
            
    ax0.set_xticks(positions0 + width * N/2)
    ax0.set_xticklabels(["x","y","z"])
    ax0.set_xlabel("Axis")
    #ax0.set_title("Repositionning")
    ax0.set_ylabel("Shift (nm)")

    ax1.set_xticks(positions1 + width * N/2)
    ax1.set_xticklabels(np.array(abmodes1)+1)
    ax1.set_xlabel("Zernike mode")
    #ax1.set_title("Aberration correction")
    ax1.set_ylabel("Amplitude (rad rms)")
    #ax1.yaxis.tick_right()
    #ax1.yaxis.set_label_position("right")

    ax1.legend()
        

def fit_fcs_file(file,pool = None,time_split = 1, first=50000):
    """Given a correction comparison file, fits the corresponding autocorrelation
    curves and extract the relevant parameters"""
    ext = comparison_file_extractor(file)
    stacks = ext["stacks"]
    npts = stacks.shape[-1]
    stacks = stacks[:,:,first:]
    filenames=ext["filenames"]
    powers = ext["powers"]
    confocal = ext["confocal"]
    
    nsplit = int(npts * 10**-6/time_split) # time in s

    
    if confocal.ndim==1:
        confocal = confocal.reshape(2,-1)
    confocal = confocal[:,first:]
    #Reverse compatibility with the gool old times
    if confocal.shape[0]==len(set(filenames))**2 and confocal.shape[1]!=1:
        confocal = confocal[::len(set(filenames))]
    spws = sted_power(powers)
    
    
    def get_conf_corrs(stks,ns,fit_method,plot=True,npool=None,names=["Before","After"],
                       supt=None,axbar=None,barunit="AR",savename=None):
        """Given a stack of confocal timetraces, splits each trace, correlates it and calculates
        the median autocorrelation"""
        corrs = [[] for w in range(stks.shape[0])]
        fits = [[] for w in range(stks.shape[0])]
        chis = [[] for w in range(stks.shape[0])]
        corrs_accepted=[]
        corrs_rejected = []
    
        for i in range(stks.shape[0]):
    
            st = stks[i]
            traces = spst(st,ns)
            
            poolnr=0
            curr_y=0
            
            for tr in traces:
                corr = multipletau.autocorrelate(tr,m=6,normalize=True,deltat=10**-3)
                x,y = corr[2:,0],corr[2:,1]
                if npool is not None:
                    curr_y+=y
                    poolnr+=1
                    if poolnr!=npool:
                        continue
                    poolnr=0
                    y = curr_y/npool
                    curr_y = 0
                pars,G = fit_method(x,y)
                chi = np.sum((y-G(x,*pars))**2)
                
                #if chi<0.5:
                
                chis[i].append(chi)
    
                fits[i].append(pars)
    
                corrs[i].append(y)
        corrs=np.asarray(corrs) 
        fit_pars=[[] for w in range(stks.shape[0])]
        for i in range(stks.shape[0]):
            chis[i] = np.asarray(chis[i])
            med_chi = np.median(chis[i])
            fit_pars[i] = np.asarray(fits[i])[:,1][chis[i]<med_chi]
            
            corrs_accepted.append( (corrs[i][ chis[i]<med_chi], 
                                    np.array(fits[i])[ chis[i]<med_chi] ) )
            
            corrs_rejected.append( (corrs[i][ chis[i]>=med_chi], 
                                    np.array(fits[i])[ chis[i]>=med_chi] ) )
            
            chis[i] = chis[i][chis[i]<med_chi]
        if plot:
            if axbar is None:
                plt.figure()
                axbar=plt.gca()
    
            axbar.boxplot(fit_pars,labels=names)
            axbar.set_ylabel(barunit)
            if supt is not None:
                axbar.set_title(supt)
                
        if plot:
            plt.figure(figsize=(12,6))
            for i in range(stks.shape[0]):
                plt.subplot(2,stks.shape[0],i+1)
                plt.title("Accepted "+names[i])
                y,par = corrs_accepted[i]
                ampl = np.mean(y[0][:10])/5
                for j in range(y.shape[0]):
                    plt.semilogx(x,y[j]+ampl*j,alpha=0.8)
                    plt.semilogx(x,G(x,*par[j])+ampl*j,"--",alpha=0.8)
                plt.xlabel("τ(ms)")
                plt.ylabel("G(τ)")
                plt.subplot(2,stks.shape[0],stks.shape[0]+i+1)
                plt.title("Rejected "+names[i])
                y,par = corrs_rejected[i]
                for j in range(y.shape[0]):
                    plt.semilogx(x,y[j]+ampl*j,alpha=0.8)
                    plt.semilogx(x,G(x,*par[j])+ampl*j,"--",alpha=0.8)
                plt.xlabel("τ(ms)")
                plt.ylabel("G(τ)")
            plt.tight_layout()
            if supt is not None:
                plt.suptitle(supt)
            if savename is not None:
                plt.savefig(savename+"_correlations"+".png")
        corrs = np.asarray(corrs)
        med_corrs = np.median(corrs,axis=1)
        return corrs,med_corrs,np.asarray(fits),fit_pars
    
    def fit_z(x,y):
        return lmfit_special(x,y,txy=0.57)
    #cconf,medconf,fitconfs = get_conf_corrs(confocal,30,npool=None)
    
    dirname = os.path.split(file)[-1].split(".")[0]+"_analysis"
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
        
    npowers = len(powers)
    u,v = find_optimal_arrangement(npowers+2)
    names_powers = ["confocal"]+[str(w) for w in powers]
    names_files = [os.path.split(w)[-1].split(".")[0][:5] for w in filenames]
    fig,axes = plt.subplots(u,v)
    axes = axes.ravel()
    all_good_fits = []
    
    c,med,fit,good_fitsConf = get_conf_corrs(confocal,nsplit,lmfit_autocorr_CH,npool=pool,
                                  names=["Before","After"],
                                  supt = "confocal",axbar = axes[0],barunit="txy (ms)",
                                  savename=dirname+"/confocal")
    
    c,med,fit,good_fitsConf = get_conf_corrs(confocal,nsplit,fit_z,npool=pool,
                                  names=["Before","After"],
                                  supt = "confocal axial fitting",axbar = axes[npowers+1],barunit="AR",
                                  savename=dirname+"/confocal_axial_fitting")
    
    for j in range(npowers):
        c2,med2,fit2,good_fits = get_conf_corrs(stacks[:,j,:],nsplit,fit_z,npool=pool,
                                      names=names_files,
                                      supt = names_powers[j+1],axbar = axes[j+1],barunit="AR",
                                      savename=dirname+"/"+str(powers[j]))
        all_good_fits.append(good_fits)

    fig.tight_layout()
    fig.savefig(dirname+"/boxplots.png")
    #if we repeated:
    ntypes = len(set(names_files))
    repeat = False
    if ntypes!=len(names_files):
        repeat=True
    pars_types = [ [] for w in range(ntypes)]
    for good_fits in all_good_fits:
        if repeat:
            n1 = len(names_files)//2
            gf = []
            for j in range(n1):
                gf.append(np.concatenate(good_fits[j::n1]))
            good_fits = gf
        for j in range(ntypes):
            pars_types[j].append(good_fits[j])
    
    spws = sted_power(powers)
    spws = np.concatenate((np.zeros(1),spws))
    pars_types = np.asarray(pars_types)
    nametypes = ["AO off","AO on"]
    
    plt.figure()
    for j in range(ntypes):
        mean = np.mean(pars_types[j],axis=1)
        std = np.std(pars_types[j],axis=1)
        
        mean = np.concatenate((np.ones(1)*4,mean))
        std = np.concatenate((np.zeros(1),std))
        plt.errorbar(spws,mean,yerr=std,label=nametypes[j])
    plt.legend()
    plt.savefig(dirname+"/fit_results.png")
    
def correlate_h5(file, mfac=16,first=50000,erase=False):
    """Computes the autocorrelation functions in a comparison_different_powers
    FCS experiment.
    Parameters:
        file: str, path to h5 file to be correlated
        mfac: int (optional) quality factor of the autocorrelation function"""
    
    if erase:
        h5f = h5py.File(file, 'a')
        if "log_autocorrelations" in h5f.keys():
            del h5f["log_autocorrelations"]
        h5f.close()
        
    h5f = h5py.File(file, 'r')

    if "log_autocorrelations" in h5f.keys() and h5f["log_autocorrelations"] is not None:
        print("Autocorrelations already computed")
        return
    
    stacks = h5f["stacks"].value
    h5f.close()
    assert(stacks.ndim==3)
    u,v,_ = stacks.shape
    correlations = [ [] for w in range(u)]
    for i in range(u):
        for j in range(v):
            print("Correlating trace "+str((i*v+ j+1))+"/"+str(u*v))
            st = stacks[i,j][first:]
            correlations[i].append(multipletau.autocorrelate(st,
                        deltat=10**-3,normalize=True,m=mfac))
    correlations = np.asarray(correlations)
    h5f = h5py.File(file, 'a')
    h5f["log_autocorrelations"] = correlations
    h5f.close()
    
def correlate_modal_h5(file, mfac=16,erase=False,first=0):
    """Computes the autocorrelation functions in a modal
    FCS experiment.
    Parameters:
        file: str, path to h5 file to be correlated
        mfac: int (optional) quality factor of the autocorrelation function
        erase: bool, optional. if True, erases previously computed correlations"""
    if erase:
        h5f = h5py.File(file, 'a')
        if "modal/log_autocorrelations" in h5f.keys():
            del h5f["modal/log_autocorrelations"]
        h5f.close()
    h5f = h5py.File(file, 'r')
    if "modal/log_autocorrelations" in h5f.keys() and h5f["modal/log_autocorrelations"] is not None\
    and h5f["modal/log_autocorrelations"].value.size>0:
        print("Autocorrelations already computed")
        return
    
    stacks = h5f["modal/log_images"].value
    h5f.close()
    assert(stacks.ndim==2)
    u,_ = stacks.shape
    correlations = [ ]
    for i in range(u):
        print("Correlating trace "+str(i+1)+"/"+str(u))
        st = stacks[i][first:]
        correlations.append(multipletau.autocorrelate(st,
                    deltat=10**-3,normalize=True,m=mfac))
    correlations = np.asarray(correlations)
    h5f = h5py.File(file, 'a')
    if "modal/log_autocorrelations" in h5f.keys():
        del h5f["modal/log_autocorrelations"]
    h5f["modal/log_autocorrelations"] = correlations
    h5f.close()


def fast_fcs_results(file,fig=None,show_legend=True,
                               show_experimental=True,fitter=None,
                                                            autocorrelate = True):
    """Plots the data from a modal FCS experiment. Computes the autocorrelations
    
    Parameters:
        file: string, filename
        fig: handle to the figure in which the data needs to be plotted. 
            if None, plots it in a new figure
        supp_metrics: list, optional. List of image quality metrics to be 
            evaluated on the dataset.
        show_legend: bool, optional. If True, displays the legend on the
            last subplot.
        show_experimental: bool, optional. If True, Displays the experimental
            (measured) data.
        names: list, optional. List of strings corresponding to the metrics
            names.
        fitter: Fitter, optional. Object of the class Fitter used to fit the 
            metric curves. If None, no fitting is applied.
    Returns: fig, handle to the figure created.
        """
    colors = ["purple","black","green","orange","brown","pink","gray","olive",
              "cyan","blue","red"]
    correlate_modal_h5(file,mfac=8)
    ext = file_extractor(file,open_stacks=False)
    log_autocorrelations = None
    log_autocorrfits = None
    
    if "log_autocorrelations" in ext:
        log_autocorrelations = ext["log_autocorrelations"]
    if "log_autocorrfits" in ext:
        log_autocorrfits = ext["log_autocorrfits"]
    
    
    if "modes" in ext:
        modes = ext['modes']
        
    else:
        modes=[-1]
        
    if "confocal" in ext:
        confocal = ext["confocal"]
    else:
        confocal = None
        
    if "P" in ext:
        P=ext["P"]
    if "xdata" in ext:
        xdata = ext['xdata']
        ydata = ext['ydata']
    else:
        xdata=np.arange(P*len(modes)).reshape(P,len(modes))
        ydata=np.ones((P,len(modes)))
        
    if "yhat" in ext:
        yhat = ext['yhat']
    else:
        yhat = ydata
        
    corrected_in=False #If True, there is a "corrected" image

    if "popt" in ext:
        popts = ext["popt"]
    else:
        popts=np.zeros((5,5))
    if "reference_aberration" in ext:
        reference_aberration = ext["reference_aberration"]
    else:
        reference_aberration = None
    xopt = None
    if "xopt" in ext:
        xopt = ext["xopt"]
    fitted_data = []
    legend = []
    
    Nz=len(modes)
    try:
        P = ext["P"]
    except:
        P = xdata.shape[0]

    x_subplot = Nz
    if fig is None:
        fig = plt.figure()
    #Prepare list containing axes
    axes=[]
    lines=[]
    
    for i in range(Nz):
        axes.append(0)
        
    for i in range(Nz):
        x=xdata[:,i]
        
        
        axes[i] = 0
        
        ax=fig.add_subplot(2,x_subplot,i+1)
        y = ydata.transpose()[i].transpose()
        yh = yhat.transpose()[i].transpose()
        
        maxval=0
        
        
        maxval = np.max(y)
        my = np.max(y,axis=0)
        y/=my
        y*=maxval
        
        p = ax.plot(reorder(x,y)[0],reorder(x,y)[1],marker="o")
        lines.append(p)
        mincorr = np.min(x)
        maxcorr=np.max(x)
        xhat = np.linspace(mincorr,maxcorr,yh.shape[0])
        p = ax.plot(xhat,maxval*yh,"--",color = colors[0])
        lines.append(p)
        if fitter is not None:
            poptf,xhf,yhf = fitter.fit(x,y)
            fitted_data.append((poptf[0],xhf,yhf))
            
        ax.axvline(popts[0,i],color='red')
        
        for j,(op,xhf,yhf) in enumerate(fitted_data):
            print("Fitted data")
            ax.plot(xhf,yhf*maxval/np.max(yhf,axis=0),'--') 
            ax.axvline(op)
            fitted_data = []
            
        ax.set_title(aberration_names[modes[i]],y=1.08)
   
        ax1 = fig.add_subplot(2,x_subplot,x_subplot+i+1)
        mode_autocorr = log_autocorrelations[i*P:(i+1)*P]
        for j in range(P):
            ax1.semilogx(mode_autocorr[j,2:,0],mode_autocorr[j,2:,1],color=colors[j%len(colors)],\
            label = str(round(x[j],2) ),alpha = 0.8)
            if log_autocorrfits is not None and len(log_autocorrfits)==len(log_autocorrelations):
                autowash = log_autocorrfits[j+i*P]
                fitval=G_corr(mode_autocorr[j,2:,0],*autowash)
                ax1.semilogx(mode_autocorr[j,2:,0],fitval,color=colors[j%len(colors)],linestyle="--")
    ax1.legend()
    
    legend = legend[:len(legend)//Nz]
    
    if show_experimental:
        tmp = ["measured","fit"]
        tmp.extend(legend)
        legend = tmp
        
    if show_legend:     
        #ax.legend(tuple(lines),tuple(legend),loc="lower right")
        
        fig.legend([x[0] for x in lines], tuple(legend), 'lower right')
    """if confocal is not None:
        ax2=fig.add_subplot(2,x_subplot,x_subplot)
        try:
            autocorr_confocal = multipletau.autocorrelate(confocal,\
                                                          deltat=10**-3,\
                                                          m=8,\
                                                          normalize=True)
            if corrected_in:
                autocorr_corrected = log_autocorrelations[-2]
            else:
                autocorr_corrected= log_autocorrelations[-P//2]
            autocorr_reference = log_autocorrelations[P//2]
            xx=mode_autocorr[0,1:,0]
            ax2.semilogx(xx,autocorr_confocal[1:,1])
            if corrected_in:
                ax2.semilogx(xx,autocorr_corrected[1:,1])
            ax2.semilogx(xx,autocorr_reference[1:,1])
            
            print("Snooop dogg")
            ax2.legend(["Confocal","After correction","Before correction"])
        except Exception as e:
            print("Snooop doggy dogg")
            print(e)"""
    return fig

def extract_modal_parameters(path,summ_name):
    infos=""
    for file in path:
        exp_name = os.path.split(file)[-1].split(".")[0]
        try:
            ext = file_extractor(file)
            P = ext["P"]
            bias = np.max(ext["xdata"])
            stacks = ext["log_images"]
            exp_time = round(stacks.shape[-1]*10**-6,2)
            print("Writing ",exp_name)
            infos+=exp_name+" exposure time "+str(exp_time)+" (s), P "+str(P)
            infos+=", Bias (rad) "+str(bias)
            infos+="\n"
        except:
            infos+=exp_name+" error\n"
    with open(summ_name,"w") as f:
        f.write(infos)

        
def plot_ac_curve(file,fig = None):
    """Method to plot an AC curve from a npy file, useful for quick inspection"""
    if file.split(".")[-1]=="SIN":
        corr = open_SIN(file)
    else:
        corr = np.load(file)
    x,y = corr[2:,0],corr[2:,1]
    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.semilogx(x,y)
    ax.axhline(0,color = "black")
    name = "".join(os.path.split(file)[-1].split(".")[:-1])
    ax.set_title("AC "+name)
    ax.set_xlabel("τ (ms)")
    ax.set_ylabel("G(τ)")

def correlate_h5_1D(file, mfac=16,first=50000,erase=False):
    """Computes the autocorrelation functions in a CH-parameter
    FCS experiment.
    Parameters:
        file: str, path to h5 file to be correlated
        mfac: int (optional) quality factor of the autocorrelation function"""
    
    if erase:
        h5f = h5py.File(file, 'a')
        if "log_autocorrelations" in h5f.keys():
            del h5f["log_autocorrelations"]
        h5f.close()
        
    h5f = h5py.File(file, 'r')

    if "log_autocorrelations" in h5f.keys() and h5f["log_autocorrelations"] is not None:
        print("Autocorrelations already computed")
        return
    
    stacks = h5f["stacks"].value
    h5f.close()
    assert(stacks.ndim==2)
    u,_ = stacks.shape
    correlations = [ ]
    for i in range(u):
        print("Correlating trace "+str(i)+"/"+str(u))
        st = stacks[i,first:]
        print(st.shape)
        correlations.append(multipletau.autocorrelate(st,
                    deltat=10**-3,normalize=True,m=mfac))
    correlations = np.asarray(correlations)
    h5f = h5py.File(file, 'a')
    h5f["log_autocorrelations"] = correlations
    h5f.close()
    
    
def display_experiment_results_cncb(file,fig=None,supp_metrics=[],show_legend=True,
                               show_experimental=True,names=[],fitter=None,
                               show_ab_diff = True):
    """Plots the data from a modal AO experiment. 
    Parameters:
        file: string, filename
        fig: handle to the figure in which the data needs to be plotted. 
            if None, plots it in a new figure
        supp_metrics: list, optional. List of image quality metrics to be 
            evaluated on the dataset.
        show_legend: bool, optional. If True, displays the legend on the
            last subplot.
        show_experimental: bool, optional. If True, Displays the experimental
            (measured) data.
        names: list, optional. List of strings corresponding to the metrics
            names.
        fitter: Fitter, optional. Object of the class Fitter used to fit the 
            metric curves. If None, no fitting is applied.
    Returns: fig, handle to the figure created.
        """
    ext = file_extractor(file)
    h5f = h5py.File(file, 'r')
    images = h5f['stacks']['stacks'][()]
    
    before = images[0]
    after = images[-1]
    images = images[1:-1]
    mname = 'measured'
    for k in h5f['metric'].keys():
        if k=='y':
            continue
        else:
            mname = k  
    h5f.close()
    
    colors_ref = ["purple","black","green","orange","brown","pink","gray","olive",
              "cyan","blue","red"]
    class CircColor(object):
        def __init__(self):
            self.colors = colors_ref
        def __getitem__(self,nb):
            return self.colors[nb%len(self.colors)]
    colors = CircColor()
    
    xdata = ext['xdata']
    ydata = ext['ydata']
    yhat = ext['yhat']
    P=xdata.shape[1]

    popts = ext["popt"]
    xopt = ext["xopt"]
    
    fitted_data = []
    legend = []
    
    Nz=xdata.shape[0]
        
    #Sanity check
    if names is not None:
        if len(names)<=len(supp_metrics):
            for j in range(len(supp_metrics)-len(names)):
                names.append("_")
            
    sup_plots=2 
        
    x_subplot = math.ceil(np.sqrt(Nz+sup_plots))
    if fig is None:
        fig = plt.figure()
    #Prepare list containing axes
    axes=[]
    for i in range(Nz+sup_plots):
        axes.append(0)
    plots=[]
    
    
    for i in range(Nz):
        axes[i] = fig.add_subplot(x_subplot,x_subplot,i+1)
        ax = axes[i]
        x = xdata[i]
        y = ydata[i]
        yh = yhat[i]
        
        maxval=0
        
        
        if show_experimental:
            maxval = np.max(y)
            my = np.max(y,axis=0)
            y/=my
            y*=maxval
            yh*=maxval/my
            p=ax.plot(x,y,marker="o")
            
            plots.append(p)
            mincorr = np.min(x)
            maxcorr=np.max(x)
            xhat = np.linspace(mincorr,maxcorr,yh.shape[0])
            p = ax.plot(xhat,yh,"--",color=colors[0])
            
            plots.append(p)
            
                
        mode_images = images[i*P:(i+1)*P]
        
        ndpl = 0 #Number of n-dimensional plots added
        for l,met in enumerate(supp_metrics):
            
            out=[]
            for j in range(P):
                out.append(met(mode_images[j]))
            out = np.asarray(out)
            
            if maxval==0:
                maxval = np.max(out)
            out = out*maxval/np.max(out,axis=0)
            
            if out.ndim==1:
                line = ax.plot(x,out,
                               marker='^',color=colors[l+ndpl+1])
                plots.append(line)
            else:
                if i==0:
                    basename = names.pop(l+ndpl)
                    print(basename)
                for kk in range(out.shape[1]):
                    line = ax.plot(x,out[:,kk],
                                   marker='^',color=colors[l+1+ndpl])
                    ndpl+=1
                    if i==0:
                        names.insert(l+ndpl+1,basename+str(ndpl))
                        print(names)
                    plots.append(line)
            
            if fitter is not None:
                poptf,xhf,yhf = fitter.fit(x,out)
                fitted_data.append((poptf[0],xhf,yhf))
        if i==0:
            legend = names[:]          
            
        if show_experimental:
            ax.axvline(xopt[i],color='red')
        
        for j,(op,xhf,yhf) in enumerate(fitted_data):
            ax.plot(xhf,yhf*maxval/np.max(yhf,axis=0),'--') 
            ax.axvline(op)
            fitted_data = []
            #,open_stacks=False
        #ax.set_title(aberration_names[modes[i]],y=1.08)
    #legend = legend[:len(legend)//Nz]
    images_comp = [before,after]
    names_im = ['before','after']
    maxval = np.array(images).max()
    for i in range(Nz,Nz+2):
        axes[i] = fig.add_subplot(x_subplot,x_subplot,i+1)
        imm = axes[i].imshow(images_comp[i-Nz],cmap = 'hot',vmax = maxval)
        axes[i].set_title(names_im[i-Nz])
        axes[i].axis('off')
    
    fig.colorbar(mappable = imm)
    if show_experimental:
        tmp = [mname,"fit"]
        tmp.extend(legend)
        legend = tmp
        
    if show_legend:     
        #ax.legend(tuple(lines),tuple(legend),loc="lower right")
        fig.legend([x[0] for x in plots], tuple(legend[:len(plots)]), 'lower right')
        #ax.legend(legend)
    fig.suptitle(file,y=1.08)
    print("End display experiment result")
    return fig