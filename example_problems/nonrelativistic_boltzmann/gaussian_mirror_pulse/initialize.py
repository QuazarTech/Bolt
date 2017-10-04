"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_f(q1, q2, p1, p2, p3, params):

    m = params.mass_particle
    k = params.boltzmann_constant

    rho_b = params.rho_background

    # Calculating the perturbed density:
    rho = rho_b + 0.01 * af.exp(-100*(q1 - 0.5)**2)

    f = rho * (m / (2 * np.pi * k))**0.5 \
            * af.exp(-m * (p1-1)**2 / (2 * k)) \

    af.eval(f)
    return (f)
