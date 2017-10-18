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
    rho = 0.01 * af.exp(-200 * (q1 - 0.5)**2)
    f   = p1 * rho

    f[:] = 0

    f[:, :, 2] = rho
    
    af.eval(f)
    return (f)
