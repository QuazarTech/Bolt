"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_f(q1, q2, p1, p2, p3, params):

    m = params.mass_particle
    k = params.boltzmann_constant

    rho  = af.select(q1<0.5, q1**0, 2/3)
    T    = af.select(q1<0.5, 2/3 * q1**0, 1/4)
    p1_b = af.select(q1<0.5, q1**0, 1.5)

    f = rho * af.sqrt(m / (2 * np.pi * k * T))**3 \
            * af.exp(-m * (p1-p1_b)**2 / (2 * k * T)) \
            * af.exp(-m * p2**2 / (2 * k * T)) \
            * af.exp(-m * p3**2 / (2 * k * T))

    af.eval(f)
    return (f)
