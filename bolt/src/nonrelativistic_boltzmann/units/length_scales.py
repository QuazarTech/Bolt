#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

import numpy as np

def gyroradius(v, B, e, m):
    r = m * v / (e * B)
    return(r)

def debye_length(n, T, e, k, eps):
    l = eps * k * T / (n * e**2)
    return(l)

def skin_depth(n, e, c, m, eps):
    l = c * np.sqrt(eps * m / (n * e**2))
    return(l)
