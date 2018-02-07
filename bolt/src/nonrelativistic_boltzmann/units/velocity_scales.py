#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

import numpy as np

def thermal_speed(T, m, k):
    v = np.sqrt(k * T / m)
    return(v)

def sound_speed(T, k, gamma):
    v = np.sqrt(gamma * k * T)
    return(v)

def alfven_velocity(B, n, m, mu):
    v_a = B / np.sqrt(n * m * mu)
    return(v_a)
