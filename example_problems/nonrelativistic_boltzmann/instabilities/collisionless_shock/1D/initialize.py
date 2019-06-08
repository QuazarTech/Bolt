"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_f(q1, q2, v1, v2, v3, params):
    
    m_e     = params.mass[0, 0]
    m_i     = params.mass[0, 1]

    k       = params.boltzmann_constant
    n       = params.n_background * q1**0

    u_be = params.u_be
    u_bi = params.u_bi
    T    = params.T_background

    f_e = n * (m_e / (2 * np.pi * k * T))**(3 / 2) \
            * 0.5 * (  af.exp(-m_e * (v1[:, 0] - u_be)**2 / (2 * k * T))
                     + af.exp(-m_e * (v1[:, 0] + u_be)**2 / (2 * k * T))
                    ) \
            * af.exp(-m_e * v2[:, 0]**2 / (2 * k * T)) \
            * af.exp(-m_e * v3[:, 0]**2 / (2 * k * T))

    f_i = n * (m_i / (2 * np.pi * k * T))**(3 / 2) \
            * 0.5 * (  af.exp(-m_i * (v1[:, 1] - u_bi)**2 / (2 * k * T))
                     + af.exp(-m_i * (v1[:, 1] + u_bi)**2 / (2 * k * T))
                    ) \
            * af.exp(-m_i * v2[:, 1]**2 / (2 * k * T)) \
            * af.exp(-m_i * v3[:, 1]**2 / (2 * k * T))

    f = af.join(1, f_e, f_i)

    af.eval(f)
    return (f)

def initialize_E(q1, q2, params):

    E1 = 0 * q1**0
    E2 = 0 * q1**0
    E3 = 0 * q1**0

    af.eval(E1, E2, E3)
    return(E1, E2, E3)

def initialize_B(q1, q2, params):
    
    # Seeding with random fluctuation:
    B1 = params.B1 * params.random_vals
    B2 = 0 * q1**0
    B3 = 0 * q1**0

    af.eval(B1, B2, B3)
    return(B1, B2, B3)
