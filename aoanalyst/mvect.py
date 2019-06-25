#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" TODO

author: J. Antonello <jacopo.antonello@dpag.ox.ac.uk>
date: Thu Nov  2 09:45:48 GMT 2017

"""

import numpy as np

from datetime import datetime
from scipy.optimize import minimize
from scipy.linalg import norm, qr
from numpy.random import normal
from scipy import ndimage

if __name__!="__main__":
    from . import msvst
else:
    import aotools
from aotools.ext.misc import entropy
from aotools.sensorless.sensorless import wavelet_decomposition,fine_wavelet_decomposition
sigma2s = []
fpr = 5e-5


def hyperspherical(u):
    # https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
    # phi1, ..., phi_n-2 [0, pi] and phi_n-1 [0, 2pi]

    assert(np.allclose(norm(u), 1))
    assert(np.nonzero(u)[0].size == u.size)

    phis = [np.arccos(u[0])]
    for i in range(1, u.size - 2):
        phis.append(np.arccos(u[i]/norm(u[i:])))

    if u[-1] >= 0:
        phis.append(np.arccos(u[-2]/norm(u[-2:])))
    else:
        phis.append(2*np.pi - np.arccos(u[-2]/norm(u[-2:])))

    phis = np.array(phis)
    assert(phis.size == u.size - 1)

    return phis


def normalised_scale_energy(ds,normalise=True):
    size = ds[0].shape[0]
    m1 = round(size*.10)
    m2 = round(size*.90)
    assert(m1 < m2)

    mv = np.array(
        [np.square(np.abs(d)).sum() for d in ds])
    mv = mv.astype(np.float)
    assert(norm(mv) > 0)
    if normalise:
        mv /= norm(mv)
        assert(np.allclose(norm(mv)**2, 1.0))
    
    return mv


def normalised_scale_entropy(ds,normalise=True):
    size = ds[0].shape[0]
    m1 = round(size*.10)
    m2 = round(size*.90)
    assert(m1 < m2)

    mv = np.array(
        [entropy(d) for d in ds])
    assert(norm(mv) > 0)
    if normalise:
        mv /= norm(mv)
        assert(np.allclose(norm(mv)**2, 1.0))
    return mv

def sobelmetric(im,horizontal = True,vertical=True):
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
    return np.sum((vert**2+hori**2))/np.sum(im)**2

def normalised_scale_edges(ds,normalise=True,horizontal=True,vertical=True):
    size = ds[0].shape[0]
    m1 = round(size*.10)
    m2 = round(size*.90)
    assert(m1 < m2)

    mv = np.array(
        [sobelmetric(d,horizontal=horizontal,vertical=vertical) for d in ds])
    assert(norm(mv) > 0)
    if normalise:
        mv /= norm(mv)
        assert(np.allclose(norm(mv)**2, 1.0))
    return mv

def compute_mvector_jacopo(img, fpr, sigma2s):
    assert((img != 0).sum() != 0)
    ajs2, djs2 = msvst.msvst(img.astype(np.float))

    if len(sigma2s) != len(djs2):
        sigma2s.clear()
        sigma2s.extend(msvst.sigma2s(len(ajs2)))

    dt = list()
    for i in range(len(djs2)):  # TODO check this range(len(djs2) - 1)?
        d = msvst.H1(djs2[i], sigma2s[i], fpr)
        dt.append(d)

    m = normalised_scale_energy(dt)
    return m

def mkpos(x):
    y=x.copy()
    y[x<0]=0
    return y
    
def msvst_denoise(x0):
    #global sigma2s
    ajs2, djs2 = msvst.msvst(x0)
    # compute level-dependent standard deviation for Normal distribution
    sigma2s = msvst.sigma2s(len(ajs2))

    # false positive rate parameter used in denoising
    fpr = 5e-5

    dt = list()
    for i in range(len(djs2) - 1):
        # apply hypothesis testing to each detail coefficient
        d = msvst.H1(djs2[i], sigma2s[i], fpr)
        dt.append(d)

    des = msvst.imsvst(ajs2,dt)
    return des

#modify mvector so that it does what it was supposed to do
def compute_mvector(img,denoise=False,djs=False,coeffs=None,normalise=True,
                    pos=False):
    global sigma2s

    assert((img != 0).sum() != 0)

    """ajs2, djs2 = msvst.msvst(img.astype(np.float),J=4)
    #ajs2 = [ajs.astype(np.float) for ajs in ajs2]
    ajs2=ajs2[1:]
    djs2=djs2[1:]
    if len(sigma2s) != len(djs2):
        sigma2s.clear()
        sigma2s.extend(msvst.sigma2s(len(ajs2)))

    dt = list()
    for i in range(len(djs2)):  # TODO check this range(len(djs2) - 1)?
        if denoise:
            d = msvst.H1(djs2[i], sigma2s[i], fpr)
        else:
            d = djs2[i]
        dt.append(d)"""
    if denoise:
        img = msvst_denoise(img)
    #!!!Hacking social
    ajs2, dt = msvst.dec(img.astype(np.float),J=5)
    dt  = dt[1:] #first coeff is caca de taureau
    if pos:
        dt = [mkpos(d) for d in dt]
    if coeffs is not None:
        dtn=[]
        for c in coeffs:
            dtn.append(dt[c])
        dt = dtn
    m = normalised_scale_energy(dt)
    return m

def test_denoising():
    from numpy.random import poisson
    from scipy.misc import ascent

    # load test image
    x0 = ascent().astype(np.float)
    x0 /= x0.max()

    # apply Poisson noise
    x0 = poisson(1000*x0)

    # compute approximation and detail coefficients
    ajs2, djs2 = msvst.msvst(x0)
    # compute level-dependent standard deviation for Normal distribution
    sigma2s = msvst.sigma2s(len(ajs2))

    # false positive rate parameter used in denoising
    fpr = 5e-3

    dt = list()
    for i in range(len(djs2) - 1):
        # apply hypothesis testing to each detail coefficient
        d = msvst.H1(djs2[i], sigma2s[i], fpr)
        dt.append(d)

    plt.subplot(1, 2, 1)
    plt.imshow(x0)
    plt.title('original')
    plt.subplot(1, 2, 2)
    des = ajs2[-1] + np.stack(dt, axis=0).sum(axis=0)
    des = msvst.imsvst(ajs2, dt)
    plt.imshow(des)
    plt.title('denoised')
    plt.show()
    
def unnormalised_vector(img):
    ajs,djs = msvst.dec(img,J=5)
    d = np.zeros(len(djs))
    for j in range(len(djs)):
        d[j] = np.sum(djs[j]**2)
    return d
def crop(img,frac=0.1):
    u,v = img.shape
    im = img[int(frac*u):int((1-frac)*u),int(frac*v):int((1-frac)*v)]
    return im
def haar(img,norm=False):
    fw = fine_wavelet_decomposition(img,wlt='haar')
    def gradient(elements):
        out=0
        for el in elements:
            out+=el**2
        return np.sqrt(out)
    fw = np.asarray([gradient(x) for x in fw])
    if norm:
        fw = normalised_scale_energy(fw)
    else:
        
        fw = [crop(x) for x in fw]
        fw = np.asarray([np.sum(x**2) for x in fw])
    return fw

def run(
        iofun, x0, maxbias, nquad, imgshape,
        overdet=5, xbiases=None, fitter_name='exp',
        h5f=None, printdb=print, abort=None):
    r"""TODO

    """

    tmp = compute_mvector(normal(size=imgshape))
    N = x0.size
    Ns = tmp.size
    Nfree = nquad.free_params(N, Ns)

    if xbiases is None:
        Nmeas = round(overdet*Nfree/(Ns - 1))
        xbiases = []
        QQ = np.eye(N)
        mul = maxbias
        while True:
            xbiases.append(mul*QQ)
            if N*len(xbiases) >= Nmeas:
                xbiases = np.hstack(xbiases)
                xbiases = xbiases[:, :Nmeas]
                break
            else:
                QQ = np.dot(QQ, qr(normal(size=(N, N)))[0])
                mul *= 4/5
    else:
        Nmeas = xbiases.shape[1]
    assert(xbiases.shape == (N, Nmeas))
    printdb('N = {}'.format(N))
    printdb('Ns = {}'.format(Ns))
    printdb('Nfree = {}'.format(Nfree))
    printdb('Nmeas = {}'.format(Nmeas))

    # collect measurements
    imgs = np.nan*np.ones((imgshape[0], imgshape[1], Nmeas))
    mvs = np.nan*np.ones((Ns, Nmeas))
    phis = np.nan*np.ones((Ns - 1, Nmeas))
    
    for i in range(xbiases.shape[1]):
        xb = xbiases[:, i]
        img = iofun(xb + x0)
        mv = compute_mvector(img)
        phi = hyperspherical(mv)

        imgs[:, :, i] = img
        mvs[:, i] = mv
        phis[:, i] = phi

        printdb('measurement {}/{}'.format(i + 1, Nmeas))

        if abort is not None and abort():
            return x0

    assert(np.all(np.isfinite(imgs)))
    assert(np.all(np.isfinite(mvs)))
    assert(np.all(np.isfinite(phis)))

    # normalise phi
    phisnorm = np.zeros_like(phis)
    for i in range(phis.shape[0]):
        p = phis[i, :]
        p -= p.min()
        p /= p.max()
        phisnorm[i, :] = p
    assert(np.all(np.isfinite(phisnorm)))

    log_err = []
    log_params = []
    p0 = nquad.make_p0(N, Ns)
    bounds = [(-maxbias, maxbias)]*N
    bounds.extend([(None, None)]*(p0.size - N))
    dt1 = datetime.now()
    optres = minimize(
            nquad.make_fun(
                xbiases, phisnorm, log_params=log_params, log_err=log_err),
            p0, method='L-BFGS-B', bounds=bounds)
    dt2 = datetime.now()
    log_err = np.array(log_err)
    log_params = np.array(log_params)
    popt = optres.x
    xopt = nquad.unpack(popt, N, 0)[0]
    printdb('log_err {}/{}'.format(log_err.max(), log_err.min()))
    printdb('success {}'.format(optres.success))
    printdb('status {}'.format(optres.status))
    printdb('message {}'.format(optres.message))
    printdb('nit {}'.format(optres.nit))
    printdb('deltat {}'.format(str(dt2 - dt1)))
    printdb('xopt {}'.format(xopt))

    if h5f:
        h5f['mvect/settings/x0'] = x0
        h5f['mvect/settings/maxbias'] = maxbias
        h5f['mvect/settings/nquad'] = nquad.__name__
        h5f['mvect/settings/imgshape'] = imgshape
        h5f['mvect/settings/overdet'] = overdet
        h5f['mvect/settings/fitter_name'] = fitter_name
        h5f['mvect/settings/N'] = N
        h5f['mvect/settings/Ns'] = Ns
        h5f['mvect/settings/Nfree'] = Nfree
        h5f['mvect/settings/Nmeas'] = Nmeas

        h5f['mvect/data/xbiases'] = xbiases
        h5f['mvect/data/imgs'] = imgs
        h5f['mvect/data/mvs'] = mvs
        h5f['mvect/data/phis'] = phis

        h5f['mvect/opt/log_err'] = log_err
        h5f['mvect/opt/log_params'] = log_params
        h5f['mvect/opt/popt'] = popt
        h5f['mvect/opt/success'] = optres.success
        h5f['mvect/opt/status'] = optres.status
        h5f['mvect/opt/message'] = optres.message
        h5f['mvect/opt/nit'] = optres.nit
        h5f['mvect/opt/deltat'] = str(dt2 - dt1)
        h5f['mvect/opt/xopt'] = xopt

    return xopt + x0
