"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_f(q1, q2, p1, p2, p3, params):

    m = params.mass_particle
    k = params.boltzmann_constant
    
    f    = q1**0 * np.sqrt(m / (2 * np.pi * 2 * k)) * af.exp(-m * p1**2 / (2 * 2 * k))
    f[:] = 0

    af.eval(f)
    return (f)
