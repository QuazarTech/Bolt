"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_f(q1, q2, v1, v2, v3, params):
    k            = params.boltzmann_constant
    n_left       = params.n_left
    v1_bulk_left = params.v1_bulk_left
    T_left       = params.T_left

    f =   q1**0 * n_left * (params.mass / (2 * np.pi * k * T_left))**(3 / 2) \
        * af.exp(-params.mass * (v1 - v1_bulk_left)**2 / (2 * k * T_left)) \
        * af.exp(-params.mass * v2**2 / (2 * k * T_left)) \
        * af.exp(-params.mass * v3**2 / (2 * k * T_left))
    
    af.eval(f)
    return (f)

def initialize_E(q1, q2, params):

    E1 = 0 * q1**0 * params.E0
    E2 = 0 * q1**0 * params.E0
    E3 = 0 * q1**0 * params.E0

    af.eval(E1, E2, E3)
    return(E1, E2, E3)

def initialize_B(q1, q2, params):

    B1 = params.B1 * q1**0
    B2 = 0 * q1**0
    B3 = 0 * q1**0

    af.eval(B1, B2, B3)
    return(B1, B2, B3)
