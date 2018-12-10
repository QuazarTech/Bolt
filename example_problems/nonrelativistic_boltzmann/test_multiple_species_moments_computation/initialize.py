"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_f(q1, q2, v1, v2, v3, params):

    m1 = params.mass[0, 0]
    m2 = params.mass[0, 1]

    k = params.boltzmann_constant

    n1 = 1 + 0.01 * af.cos(2 * np.pi * q1)
    n2 = 1 + 0.02 * af.cos(4 * np.pi * q1)

    T1 = 1   + 0.02 * af.sin(2 * np.pi * q1)
    T2 = 100 + 0.01 * af.sin(4 * np.pi * q1)

    f1 = n1 * (1 / (2 * np.pi * k * T1))**(1 / 2) \
            * af.exp(-m1 * v1[:, 0]**2 / (2 * k * T1)) \

    f2 = n2 * (1 / (2 * np.pi * k * T2))**(1 / 2) \
            * af.exp(-m2 * v1[:, 1]**2 / (2 * k * T2)) \

    f = af.join(1, f1, f2)
    af.eval(f)
    return (f)
