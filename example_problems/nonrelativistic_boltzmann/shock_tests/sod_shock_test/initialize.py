"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np


def initialize_f(q1, q2, p1, p2, p3, params):
    import arrayfire as af

    m = params.mass_particle
    k = params.boltzmann_constant

    rho = af.select(q1<0.5, 1, 0.125)
    T   = af.select(q1<0.5, 1, 0.8)

    f = rho * np.sqrt(m / (2 * np.pi * k * T)) \
            * af.exp(-m * p1**2 / (2 * k * T))

    af.eval(f)
    return (f)
