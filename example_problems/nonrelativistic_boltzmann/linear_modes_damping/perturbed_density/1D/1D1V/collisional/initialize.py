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
    T_b   = params.temperature_background

    p1_bulk = params.p1_bulk_background

    pert_real = params.pert_real
    pert_imag = params.pert_imag

    k_q1 = params.k_q1

    # Calculating the perturbed density:
    rho = rho_b + (  pert_real * af.cos(k_q1 * q1)
                   - pert_imag * af.sin(k_q1 * q1)
                  )

    f = rho * np.sqrt(m / (2 * np.pi * k * T_b)) \
            * af.exp(-m * (p1 - p1_bulk)**2 / (2 * k * T_b))

    af.eval(f)
    return (f)
