"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np


def initialize_f(q1, q2, p1, p2, p3, params):

    m = params.mass_particle
    k = params.boltzmann_constant

    rho = (af.abs(q2) < 0.25) * 2 + (af.abs(q2) > 0.25)

    p1_bulk = (af.abs(q2) < 0.25) * 0.5 - (af.abs(q2) > 0.25) * 0.5

    T = (1 / 3) * (2.5 / rho)

    f = rho * (m / (2 * np.pi * k * T))**(3 / 2) * \
        af.exp(-m * (p1 - p1_bulk)**2 / (2 * k * T)) * \
        af.exp(-m * (p2)**2 / (2 * k * T)) * \
        af.exp(-m * (p3)**2 / (2 * k * T))

    af.eval(f)
    return (f)
