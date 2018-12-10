"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_f(q1, q2, v1, v2, v3, params):

    m = params.mass
    k = params.boltzmann_constant

    n_b = params.density_background
    T_b = params.temperature_background

    n = n_b + 0 * q1**0
    T = af.select((q1**2 + q2**2) < 0.01,
    			  100 * T_b * q1**0,
    			  T_b
    			 )

    f = n * (m / (2 * np.pi * k * T)) \
          * af.exp(-m * v1**2 / (2 * k * T)) \
          * af.exp(-m * v2**2 / (2 * k * T))

    af.eval(f)
    return (f)

def initialize_E(q1, q2, params):

    E1 = 0 * q1**0
    E2 = 0 * q1**0
    E3 = 0 * q1**0

    return(E1, E2, E3)

def initialize_B(q1, q2, params):

    B1 = params.B1 * q1**0
    B2 = params.B1 * q1**0
    B3 = 0 * q1**0

    af.eval(B1, B2, B3)
    return(B1, B2, B3)
