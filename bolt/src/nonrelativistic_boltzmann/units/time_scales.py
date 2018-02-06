#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

import numpy as np

def plasma_frequency(n, e, m, eps):
    omega = np.sqrt(n * e**2 / (m * eps))
    return(omega)

def gyrofrequency(B, e, m):
    omega = e * B / m
    return(omega)

def alfven_time(l, B, n, m, mu):
    v_a   = B / np.sqrt(n * m * mu)
    tau_a = l / v_a
    return(tau_a)
