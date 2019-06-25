#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:26:05 2019

@author: aurelien
"""

import numpy as np
import h5py

experiment_class = ["modal","fcs","correlation","FCS_calibration"]

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

def modal_fcs_extractor(file):
    """Does not extract the stacks as they are useless in most cases and can
    be very large"""
    h5f = h5py.File(file, 'r')
    out = {} 
    for key in h5f['modal/'].keys():
        if key!="log_images":
            out[key] = h5f['modal/'][key].value
    return out
