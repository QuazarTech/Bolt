"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np
from domain import q1_end, q1_start, N_q1, \
                   q2_end, q2_start, N_q2

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

    n = n_b + af.exp(-(q1 - 0.5)**2 - (q2 - 0.5)**2)

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

def initialize_A3_B3(q1, q2, params):

    amp_real = params.amp_real
    amp_imag = params.amp_imag

    k_q1 = params.k_q1
    k_q2 = params.k_q2

    A3 =   amp_real * af.cos(k_q1 * q1 + k_q2 * q2) * params.B0 \
         - amp_imag * af.sin(k_q1 * q1 + k_q2 * q2) * params.B0
    B3 = params.B0 * q1**0

    af.eval(A3, B3)
    return(A3, B3)

# Alternatively, trying this out(HACKY VERSION)
# def initialize_B(q1, q2, params):
#     amp_real = params.amp_real
#     amp_imag = params.amp_imag

#     k_q1 = params.k_q1
#     k_q2 = params.k_q2

#     A3 =   amp_real * af.cos(k_q1 * q1 + k_q2 * q2) * params.B0 \
#          - amp_imag * af.sin(k_q1 * q1 + k_q2 * q2) * params.B0

#     A3_plus_q2 = af.shift(A3, 0, 0,  0, -1)
#     A3_plus_q1 = af.shift(A3, 0, 0, -1,  0)

#     dq1 = (q1_end - q1_start) / N_q1
#     dq2 = (q2_end - q2_start) / N_q2

#     B1 =  (A3_plus_q2 - A3) / dq2 #  dA3_dq2
#     B2 = -(A3_plus_q1 - A3) / dq1 # -dA3_dq1
#     B3 = params.B0 * q1**0

#     af.eval(B1, B2, B3)
#     return(B1, B2, B3)
