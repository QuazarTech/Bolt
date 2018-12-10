#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

import numpy as np

def thermal_speed(T, m, k):
    """
    Returns the plasma frequency 

    Parameters:
    -----------
    T: mean temperature of the plasma

    m: mass of the charged species considered.

    k: Boltzmann constant
    """
    v = np.sqrt(k * T / m)
    return(v)

def sound_speed(T, k, gamma):
    """
    Returns the sound

    Parameters:
    -----------
    T: mean temperature of the plasma

    k: Boltzmann constant

    gamma: adiabatic constant
    """
    v = np.sqrt(gamma * k * T)
    return(v)

def alfven_velocity(B, n, m, mu):
    """
    Returns the alfven velocity 

    Parameters:
    -----------
    B: magnitude of magnetic field
    
    n: mean number density of the plasma

    m: mass of the charged species considered.

    mu: magnetic permeability
    """
    v_a = B / np.sqrt(n * m * mu)
    return(v_a)
