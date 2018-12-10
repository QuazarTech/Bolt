#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import arrayfire as af

def calculate_p_center(p1_start, p2_start, p3_start,
                       N_p1, N_p2, N_p3,
                       dp1, dp2, dp3, 
                      ):
    """
    Initializes the cannonical variables p1, p2 and p3 using a centered
    formulation.
    """
    p1_center = af.constant(0, N_p1 * N_p2 * N_p3, len(p1_start), dtype = af.Dtype.f64)
    p2_center = af.constant(0, N_p1 * N_p2 * N_p3, len(p2_start), dtype = af.Dtype.f64)
    p3_center = af.constant(0, N_p1 * N_p2 * N_p3, len(p3_start), dtype = af.Dtype.f64)

    # Assigning for each species:
    for i in range(len(p1_start)):

        p1 = p1_start[i] + (0.5 + np.arange(N_p1)) * dp1[i]
        p2 = p2_start[i] + (0.5 + np.arange(N_p2)) * dp2[i]
        p3 = p3_start[i] + (0.5 + np.arange(N_p3)) * dp3[i]
        
        p2_center[:, i] = af.flat(af.to_array(np.meshgrid(p2, p1, p3)[0]))
        p1_center[:, i] = af.flat(af.to_array(np.meshgrid(p2, p1, p3)[1]))
        p3_center[:, i] = af.flat(af.to_array(np.meshgrid(p2, p1, p3)[2]))

    af.eval(p1_center, p2_center, p3_center)
    return (p1_center, p2_center, p3_center)

def calculate_p_left(p1_start, p2_start, p3_start,
                     N_p1, N_p2, N_p3,
                     dp1, dp2, dp3, 
                    ):
    p1_left = af.constant(0, N_p1 * N_p2 * N_p3, len(p1_start), dtype = af.Dtype.f64)
    p2_left = af.constant(0, N_p1 * N_p2 * N_p3, len(p2_start), dtype = af.Dtype.f64)
    p3_left = af.constant(0, N_p1 * N_p2 * N_p3, len(p3_start), dtype = af.Dtype.f64)

    # Assigning for each species:
    for i in range(len(p1_start)):

        p1 = p1_start[i]  + np.arange(N_p1)        * dp1[i] 
        p2 = p2_start[i] + (0.5 + np.arange(N_p2)) * dp2[i]
        p3 = p3_start[i] + (0.5 + np.arange(N_p3)) * dp3[i]
        
        p2_left[:, i] = af.flat(af.to_array(np.meshgrid(p2, p1, p3)[0]))
        p1_left[:, i] = af.flat(af.to_array(np.meshgrid(p2, p1, p3)[1]))
        p3_left[:, i] = af.flat(af.to_array(np.meshgrid(p2, p1, p3)[2]))

    af.eval(p1_left, p2_left, p3_left)
    return (p1_left, p2_left, p3_left)

def calculate_p_bottom(p1_start, p2_start, p3_start,
                       N_p1, N_p2, N_p3,
                       dp1, dp2, dp3, 
                      ):
    p1_bottom = af.constant(0, N_p1 * N_p2 * N_p3, len(p1_start), dtype = af.Dtype.f64)
    p2_bottom = af.constant(0, N_p1 * N_p2 * N_p3, len(p2_start), dtype = af.Dtype.f64)
    p3_bottom = af.constant(0, N_p1 * N_p2 * N_p3, len(p3_start), dtype = af.Dtype.f64)

    # Assigning for each species:
    for i in range(len(p1_start)):

        p1 = p1_start[i] + (0.5 + np.arange(N_p1)) * dp1[i]
        p2 = p2_start[i] + np.arange(N_p2)         * dp2[i]
        p3 = p3_start[i] + (0.5 + np.arange(N_p3)) * dp3[i]
        
        p2_bottom[:, i] = af.flat(af.to_array(np.meshgrid(p2, p1, p3)[0]))
        p1_bottom[:, i] = af.flat(af.to_array(np.meshgrid(p2, p1, p3)[1]))
        p3_bottom[:, i] = af.flat(af.to_array(np.meshgrid(p2, p1, p3)[2]))

    af.eval(p1_bottom, p2_bottom, p3_bottom)
    return (p1_bottom, p2_bottom, p3_bottom)

def calculate_p_back(p1_start, p2_start, p3_start,
                     N_p1, N_p2, N_p3,
                     dp1, dp2, dp3, 
                    ):
    p2_back = af.constant(0, N_p1 * N_p2 * N_p3, len(p1_start), dtype = af.Dtype.f64)
    p1_back = af.constant(0, N_p1 * N_p2 * N_p3, len(p2_start), dtype = af.Dtype.f64)
    p3_back = af.constant(0, N_p1 * N_p2 * N_p3, len(p3_start), dtype = af.Dtype.f64)

    # Assigning for each species:
    for i in range(len(p1_start)):

        p1 = p1_start[i] + (0.5 + np.arange(N_p1)) * dp1[i]
        p2 = p2_start[i] + (0.5 + np.arange(N_p2)) * dp2[i]
        p3 = p3_start[i] + np.arange(N_p3)         * dp3[i]
        
        p2_back[:, i] = af.flat(af.to_array(np.meshgrid(p2, p1, p3)[0]))
        p1_back[:, i] = af.flat(af.to_array(np.meshgrid(p2, p1, p3)[1]))
        p3_back[:, i] = af.flat(af.to_array(np.meshgrid(p2, p1, p3)[2]))

    af.eval(p1_back, p2_back, p3_back)
    return (p1_back, p2_back, p3_back)
