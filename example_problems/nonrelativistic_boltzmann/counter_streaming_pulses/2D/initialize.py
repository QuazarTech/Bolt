"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_f(q1, q2, v1, v2, v3, params):

    m_e = params.mass[0, 0]
    m_p = params.mass[0, 1]

    k = params.boltzmann_constant

    n_b = params.density_background
    T_b = params.temperature_background

    k = params.boltzmann_constant

    v1_bulk_electron = params.v1_bulk_electron
    v1_bulk_positron = params.v1_bulk_positron

    v2_bulk_electron = params.v2_bulk_electron
    v2_bulk_positron = params.v2_bulk_positron

    n = n_b + 0.1 * af.exp(-10 * (q1 - 5)**2 -10 * (q2 - 5)**2)

    f_e = n * (m_e / (2 * np.pi * k * T_b)) \
            * af.exp(-m_e * (v1[:, 0] - v1_bulk_electron)**2 / (2 * k * T_b)) \
            * af.exp(-m_e * (v2[:, 0] - v2_bulk_electron)**2 / (2 * k * T_b))

    f_p = n * (m_p / (2 * np.pi * k * T_b)) \
            * af.exp(-m_p * (v1[:, 1] - v1_bulk_positron)**2 / (2 * k * T_b)) \
            * af.exp(-m_p * (v2[:, 1] - v2_bulk_positron)**2 / (2 * k * T_b))

    f = af.join(1, f_e, f_p)

    af.eval(f)
    return (f)

def initialize_E(q1, q2, params):

    E1 = 0 * q1**0
    E2 = 0 * q1**0
    E3 = 0 * q1**0

    return(E1, E2, E3)

def initialize_B(q1, q2, params):

    B1 = params.B1 * q1**0
    B2 = 0 * q1**0
    B3 = 0 * q1**0

    af.eval(B1, B2, B3)
    return(B1, B2, B3)
