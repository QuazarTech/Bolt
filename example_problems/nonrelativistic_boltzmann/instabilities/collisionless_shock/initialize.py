"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_f(q1, q2, p1, p2, p3, params):
    
    f = 0 * q1**0 * p1**0
    af.eval(f)
    return (f)

def initialize_E(q1, q2, params):

    E1 = 0 * q1**0 * params.E0
    E2 = 0 * q1**0 * params.E0
    E3 = 0 * q1**0 * params.E0

    af.eval(E1, E2, E3)
    return(E1, E2, E3)

def initialize_B(q1, q2, params):

    B1 =   np.sqrt(2 * params.n_left * params.T_left / params.plasma_beta) \
         * params.B0 * q1**0
    B2 = 0 * q1**0 * params.B0
    B3 = 0 * q1**0 * params.B0

    af.eval(B1, B2, B3)
    return(B1, B2, B3)
