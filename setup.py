#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:43:08 2019

@author: aurelien
"""

from setuptools import setup, find_packages

    
setup(name='aoanalyst',
      version='0.1',
      description='Package for analysis of AO data',
      long_description = "GUI for AO data visualisation and analysis",
      url='',
      packages = find_packages(),
      author='Aurelien Barbotin',
      author_email='aurelien.barbotin@dtc.ox.ac.uk',
      package_data = {},
      include_package_data = True,
      license='MIT',
      zip_safe=False)
