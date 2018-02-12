#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

import numpy as np

def gyroradius(v, B, e, m):
    """
    Returns the gyroradius

    Parameters:
    -----------
    v: gyrovelocity

    B: magnitude of magnetic field
    
    e: magnitude of electric charge.

    m: mass of the charged species considered.
    """
    r = m * v / (e * B)
    return(r)

def debye_length(n, T, e, k, eps):
    """
    Returns the plasma frequency 

    Parameters:
    -----------
    n: mean number density of the plasma

    T: mean temperature of the plasma

    e: magnitude of electric charge.

    k: Boltzmann constant

    eps: permittivity
    """
    l = eps * k * T / (n * e**2)
    return(l)

def skin_depth(n, e, c, m, eps):
    """
    Returns the plasma frequency 

    Parameters:
    -----------
    n: mean number density of the plasma

    e: magnitude of electric charge.

    c: Speed of light

    m: mass of the charged species considered.

    eps: permittivity
    """
    l = c * np.sqrt(eps * m / (n * e**2))
    return(l)
