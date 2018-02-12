#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

import numpy as np

def plasma_frequency(n, e, m, eps):
    """
    Returns the plasma frequency 

    Parameters:
    -----------
    n: mean number density of the plasma

    e: magnitude of electric charge.

    m: mass of the charged species considered.

    eps: permittivity
    """
    omega = np.sqrt(n * e**2 / (m * eps))
    return(omega)

def cyclotron_frequency(B, e, m):
    """
    Returns the cyclotron frequency 

    Parameters:
    -----------
    B: magnitude of magnetic field

    e: magnitude of electric charge.

    m: mass of the charged species considered.
    """
    omega = e * B / m
    return(omega)

def alfven_time(l, B, n, m, mu):
    """
    Returns the Alfven time 

    Parameters:
    -----------
    l:  characteristic length scale of the system.
    
    B:  magnitude of magnetic field
    
    n:  mean number density of the plasma
    
    m:  mass of the charged species considered.
    
    mu: magnetic permeability
    """
    v_a   = B / np.sqrt(n * m * mu)
    tau_a = l / v_a
    return(tau_a)
