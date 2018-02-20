"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np


def initialize_f(q1, q2, v1, v2, v3, params):

    m_e = params.mass[0, 0]
    m_i = params.mass[0, 1]

    k = params.boltzmann_constant

    n_b_e = params.n_background_e
    T_b_e = params.temperature_background_e

    n_b_i = params.n_background_i
    T_b_i = params.temperature_background_i

    n_e = n_b_e + 0 * q1
    n_i = n_b_i + params.alpha * af.cos(q1)
    T_e = T_b_e
    T_i = T_b_i

    f_e = n_e * np.sqrt(1 / (2 * np.pi)) * af.sqrt(m_e * T_i/m_i * T_e) \
              * af.exp(-0.5 * (m_e * T_i/m_i * T_e) * (v1[:, 0])**2)

    f_i = n_i * np.sqrt(1 / (2 * np.pi)) \
              * af.exp(-0.5 * (v1[:, 1])**2)

    f = af.join(1, f_e, f_i)
    
    af.eval(f)
    return (f)
