"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_f(q1, q2, v1, v2, v3, params):
    
    m = params.mass
    k = params.boltzmann_constant

    n_b  = params.density_background
    T_b1 = params.temperature_background_1
    T_b2 = params.temperature_background_2
    T_b3 = params.temperature_background_3

    k = params.boltzmann_constant

    n = n_b + params.amplitude * af.cos(params.k_q1 * q1)

    f1 = n * (m[0, 0] / (2 * np.pi * k * T_b1))**(1 / 2) \
           * af.exp(-m[0, 0] * v1[:, 0]**2 / (2 * k * T_b1))

    f2 = n * (m[0, 1] / (2 * np.pi * k * T_b2))**(1 / 2) \
           * af.exp(-m[0, 1] * v1[:, 1]**2 / (2 * k * T_b2))

    f3 = n * (m[0, 2] / (2 * np.pi * k * T_b3))**(1 / 2) \
           * af.exp(-m[0, 2] * v1[:, 2]**2 / (2 * k * T_b3))

    f = af.join(1, f1, f2, f3)

    af.eval(f)
    return (f)
