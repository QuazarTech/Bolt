#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import arrayfire as af

def calculate_p_center(p1_start, p2_start, p3_start,
                       N_p1, N_p2, N_p3,
                       dp1, dp2, dp3, 
                       N_species = 1
                      ):
    """
    Initializes the cannonical variables p1, p2 and p3 using a centered
    formulation.
    """
    p1_center = p1_start + (0.5 + np.arange(N_p1)) * dp1
    p2_center = p2_start + (0.5 + np.arange(N_p2)) * dp2
    p3_center = p3_start + (0.5 + np.arange(N_p3)) * dp3
    
    p2_center, p1_center, p3_center = np.meshgrid(p2_center,
                                                  p1_center,
                                                  p3_center
                                                 )

    # Flattening the arrays:
    p1_center = af.flat(af.to_array(p1_center))
    p2_center = af.flat(af.to_array(p2_center))
    p3_center = af.flat(af.to_array(p3_center))

    p1_center = af.tile(p1_center, 1, N_species)
    p2_center = af.tile(p2_center, 1, N_species)
    p3_center = af.tile(p3_center, 1, N_species)

    af.eval(p1_center, p2_center, p3_center)
    return (p1_center, p2_center, p3_center)

def calculate_p_left(p1_start, p2_start, p3_start,
                     N_p1, N_p2, N_p3,
                     dp1, dp2, dp3, 
                     N_species = 1
                    ):

    p1_left   = p1_start + np.arange(N_p1) * dp1
    p2_center = p2_start + (0.5 + np.arange(N_p2)) * dp2
    p3_center = p3_start + (0.5 + np.arange(N_p3)) * dp3

    p2_left, p1_left, p3_left = np.meshgrid(p2_center,
                                            p1_left,
                                            p3_center
                                           )

    # Flattening the arrays:
    p1_left = af.flat(af.to_array(p1_left))
    p2_left = af.flat(af.to_array(p2_left))
    p3_left = af.flat(af.to_array(p3_left))

    p1_left = af.tile(p1_left, 1, N_species)
    p2_left = af.tile(p2_left, 1, N_species)
    p3_left = af.tile(p3_left, 1, N_species)

    af.eval(p1_left, p2_left, p3_left)
    return (p1_left, p2_left, p3_left)

def calculate_p_bottom(p1_start, p2_start, p3_start,
                       N_p1, N_p2, N_p3,
                       dp1, dp2, dp3, 
                       N_species = 1
                      ):

    p1_center = p1_start + (0.5 + np.arange(N_p1)) * dp1
    p2_bottom = p2_start + np.arange(N_p2) * dp2
    p3_center = p3_start + (0.5 + np.arange(N_p3)) * dp3

    p2_bottom, p1_bottom, p3_bottom = np.meshgrid(p2_bottom,
                                                  p1_center,
                                                  p3_center
                                                 )

    # Flattening the arrays:
    p1_bottom = af.flat(af.to_array(p1_bottom))
    p2_bottom = af.flat(af.to_array(p2_bottom))
    p3_bottom = af.flat(af.to_array(p3_bottom))

    p1_bottom = af.tile(p1_bottom, 1, N_species)
    p2_bottom = af.tile(p2_bottom, 1, N_species)
    p3_bottom = af.tile(p3_bottom, 1, N_species)

    af.eval(p1_bottom, p2_bottom, p3_bottom)
    return (p1_bottom, p2_bottom, p3_bottom)

def calculate_p_back(p1_start, p2_start, p3_start,
                     N_p1, N_p2, N_p3,
                     dp1, dp2, dp3, 
                     N_species = 1
                    ):

    p1_center = p1_start + (0.5 + np.arange(N_p1)) * dp1
    p2_center = p2_start + (0.5 + np.arange(N_p2)) * dp2
    p3_back   = p3_start + np.arange(N_p3) * dp3

    p2_back, p1_back, p3_back = np.meshgrid(p2_center,
                                            p1_center,
                                            p3_back
                                           )

    # Flattening the arrays:
    p1_back = af.flat(af.to_array(p1_back))
    p2_back = af.flat(af.to_array(p2_back))
    p3_back = af.flat(af.to_array(p3_back))
    
    p1_back = af.tile(p1_back, 1, N_species)
    p2_back = af.tile(p2_back, 1, N_species)
    p3_back = af.tile(p3_back, 1, N_species)

    af.eval(p1_back, p2_back, p3_back)
    return (p1_back, p2_back, p3_back)
