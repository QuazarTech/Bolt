"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_f(q1, q2, v1, v2, v3, params):

    n0      = params.density
    v1_bulk = params.v1_bulk
    beta0   = params.beta0

    f =   (n0 / (np.pi * beta0)**(3 / 2)) \
        * af.exp(-(v1 - v1_bulk)**2 / beta0) \
        * af.exp(-v2**2 / beta0) \
        * af.exp(-v3**2 / beta0)

    af.eval(f)
    return (f)

def initialize_B(q1, q2, params):

    B1 = 1 * q1**0
    B2 = 0 * q1**0
    B3 = 0 * q1**0

    af.eval(B1, B2, B3)
    return(B1, B2, B3)
