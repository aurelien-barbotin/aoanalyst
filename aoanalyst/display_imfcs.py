#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 13:30:12 2021

@author: aurelien
"""

from PyImFCS.hover_plot import multiplot_stack
from PyImFCS.class_imFCS import StackFCS

import matplotlib.pyplot as plt

def interactive_plot_h5(path, fig = None, nsum = 2):
    if fig is None:
        fig,axes = plt.subplots(2,4,figsize = (10,7))
    stack = StackFCS(path)
    stack.load()
    multiplot_stack(stack,nsum)