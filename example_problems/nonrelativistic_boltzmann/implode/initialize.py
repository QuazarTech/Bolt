"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_f(q1, q2, v1, v2, v3, params):

    m = params.mass
    k = params.boltzmann_constant

    # Calculating the perturbed density:
    n = af.select(q1+q2>0.15, q1**0, 0.125)
    T = af.select(q1+q2>0.15, q1**0, 0.373)
 
    f = n * (m / (2 * np.pi * k * T)) \
          * af.exp(-m * v1**2 / (2 * k * T)) \
          * af.exp(-m * v2**2 / (2 * k * T))
    
    af.eval(f)
    return (f)
