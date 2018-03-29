"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_f(q1, q2, v1, v2, v3, params):
    
    m = params.mass
    k = params.boltzmann_constant

    n = af.select(q1<0.5, q1**0, 0.125)
    T = af.select(q1<0.5, q1**0, 0.8)

    f = n * (m / (2 * np.pi * k * T))**(3 / 2) \
          * af.exp(-m * v1**2 / (2 * k * T)) \
          * af.exp(-m * v2**2 / (2 * k * T)) \
          * af.exp(-m * v3**2 / (2 * k * T))

    af.eval(f)
    return (f)

def initialize_E(q1, q2, params):

    E1 = 0 * q1**0
    E2 = 0 * q1**0
    E3 = 0 * q1**0

    return(E1, E2, E3)

def initialize_B(q1, q2, params):

    B1 = 0 * q1**0
    B2 = af.select(q1<0.5, params.B0 * 1 * q1**0, -params.B0)
    B3 = 0 * q1**0

    af.eval(B1, B2, B3)
    return(B1, B2, B3)
