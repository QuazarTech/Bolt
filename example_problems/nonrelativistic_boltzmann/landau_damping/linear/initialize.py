"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_f(q1, q2, v1, v2, v3, params):

    m = params.mass
    k = params.boltzmann_constant

    n_b = params.n_background
    T_b = params.temperature_background

    n = n_b + params.alpha * af.cos(0.5 * q1)
    T = T_b + 0 * q1

    f = n * (m / (2 * np.pi * k * T))**(1 / 2) \
          * af.exp(-m * v1**2 / (2 * k * T)) \
    
    af.eval(f)
    return (f)
