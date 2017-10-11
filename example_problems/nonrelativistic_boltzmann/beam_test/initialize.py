"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_f(q1, q2, p1, p2, p3, params):

    m = params.mass_particle
    k = params.boltzmann_constant
    
    f    = q1**0 * np.sqrt(m / (2 * np.pi * 1.5 * k)) * af.exp(-m * p1**2 / (2 * 1.5 * k))
    f[:] = 0
    
    # f    = af.moddims(f, 134*134, 4, 4)
    
    # f[:, 9, 9] = 1
    # f          = af.moddims(f, 134, 134, 4*4)

    af.eval(f)
    return (f)
