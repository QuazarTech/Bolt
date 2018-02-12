"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_f(q1, q2, p1, p2, p3, params):

    m = params.mass
    k = params.boltzmann_constant

    f = rho * (m / (2 * np.pi * k * T_b))**(3 / 2) \
            * af.exp(-m * (p1 - p1_bulk)**2 / (2 * k * T_b)) \
            * af.exp(-m * (p2 - p2_bulk)**2 / (2 * k * T_b)) \
            * af.exp(-m * (p3 - p3_bulk)**2 / (2 * k * T_b))

    af.eval(f)
    return (f)

def initialize_B(q1, q2, params):
    B1 = 1 * q1**0
    B2 = 0 * q1**0
    B3 = 0 * q1**0

    return(B1, B2, B3)
