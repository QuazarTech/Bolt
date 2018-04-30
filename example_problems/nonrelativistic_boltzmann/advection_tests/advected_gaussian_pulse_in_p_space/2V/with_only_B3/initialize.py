"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_f(q1, q2, v1, v2, v3, params):

    m = params.mass
    k = params.boltzmann_constant

    rho_b = params.rho_background
    T_b   = params.temperature_background

    v1_bulk = params.v1_bulk_background
    v2_bulk = params.v2_bulk_background

    rho = rho_b + 0 * q1
    f   = rho * (m / (2 * np.pi * k * T_b)) \
              * af.exp(-m * (v1 - v1_bulk)**2 / (2 * k * T_b)) \
              * af.exp(-m * (v2 - v2_bulk)**2 / (2 * k * T_b)) \

    af.eval(f)
    return (f)

def initialize_E(q1, q2, t):
    
    E1 = 0 * q1**0
    E2 = 0 * q1**0
    E3 = 0 * q1**0

    return(E1, E2, E3)

def initialize_B(q1, q2, t):

    B1 = 0. * q1**0
    B2 = 0. * q1**0 
    B3 = 5. * q1**0

    return(B1, B2, B3)
