#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import arrayfire as af

def calculate_k(N_q1, N_q2, dq1, dq2):
    """
    Initializes the wave numbers k_q1 and k_q2 which will be 
    used when solving in fourier space.
    """
    k_q1 = 2 * np.pi * np.fft.fftfreq(N_q1, dq1)
    k_q2 = 2 * np.pi * np.fft.fftfreq(N_q2, dq2)

    k_q2, k_q1 = np.meshgrid(k_q2, k_q1)

    k_q1 = af.to_array(k_q1)
    k_q2 = af.to_array(k_q2)

    k_q1 = af.reorder(k_q1, 2, 3, 0, 1)
    k_q2 = af.reorder(k_q2, 2, 3, 0, 1)

    af.eval(k_q1, k_q2)
    return(k_q1, k_q2)
