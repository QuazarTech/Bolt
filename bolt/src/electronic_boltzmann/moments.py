#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

import numpy as np

from bolt.src.utils.integral_over_v import integral_over_v

import params

def density(f, v1, v2, v3, integral_measure):
    return(integral_over_v(f * v1**(4./(2.*np.pi*params.h_bar)), integral_measure))

def j_x(f, v1, v2, v3, integral_measure):
    return(integral_over_v(f * v1**(4./(2.*np.pi*params.h_bar)), integral_measure))

def j_y(f, v1, v2, v3, integral_measure):
    return(integral_over_v(f * v2**(4./(2.*np.pi*params.h_bar)), integral_measure))
