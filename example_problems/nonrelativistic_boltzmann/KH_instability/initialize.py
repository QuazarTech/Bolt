"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np


def initialize_f(q1, q2, p1, p2, p3, params):

    m = params.mass_particle
    k = params.boltzmann_constant

    q2_minus = -0.25
    q2_plus = 0.25

    regulator = 50  # larger value makes the transition sharper

    amplitude_p1 = 0.5

    rho = 0.5 * (np.tanh(( q2 - q2_minus)*regulator) -
                 np.tanh(( q2 - q2_plus)*regulator)) + 1

    p1_bulk = amplitude_p1 * \
              (af.tanh(( q2 - q2_minus)*regulator) -
               af.tanh(( q2 - q2_plus )*regulator) - 1)

    T = (2.5 / rho)

    f = rho * (m / (2 * np.pi * k * T))**(3 / 2) * \
        af.exp(-m * (p1 - p1_bulk)**2 / (2 * k * T)) * \
        af.exp(-m * (p2)**2 / (2 * k * T)) * \
        af.exp(-m * (p3)**2 / (2 * k * T))

    af.eval(f)
    return (f)
