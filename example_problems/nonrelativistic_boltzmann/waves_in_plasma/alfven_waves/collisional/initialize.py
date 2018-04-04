"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_f(q1, q2, v1, v2, v3, params):
    
    m = params.mass
    k = params.boltzmann_constant

    n_b = params.density_background
    T_b = params.temperature_background

    k = params.boltzmann_constant

    v1_bulk   = 0

    # Assigning separate bulk velocities
    v2_bulk_p =   params.amplitude * 0.29043147224897364 * af.cos(params.k_q1 * q1) \
    			- params.amplitude * 0.005286752034648054 * af.sin(params.k_q1 * q1)

    v2_bulk_e =   params.amplitude * 0.29043147224897375 * af.cos(params.k_q1 * q1) \
                - params.amplitude * 0.010063251560685632 * af.sin(params.k_q1 * q1)
    
    v3_bulk_p =   params.amplitude * 0.3390205750548746 * af.cos(params.k_q1 * q1) \
    			- params.amplitude * 0.002045634873296759* af.sin(params.k_q1 * q1)

    v3_bulk_e =   params.amplitude * 0.3391286917903521 * af.cos(params.k_q1 * q1) \
                - params.amplitude * -0.0020456348732508928 * af.sin(params.k_q1 * q1)

    n = n_b + 0 * q1**0

    f_e = n * (m[0, 0] / (2 * np.pi * k * T_b))**(3 / 2) \
            * af.exp(-m[0, 0] * (v1[:, 0] - v1_bulk)**2 / (2 * k * T_b)) \
            * af.exp(-m[0, 0] * (v2[:, 0] - v2_bulk_e)**2 / (2 * k * T_b)) \
            * af.exp(-m[0, 0] * (v3[:, 0] - v3_bulk_e)**2 / (2 * k * T_b))

    f_p = n * (m[0, 1] / (2 * np.pi * k * T_b))**(3 / 2) \
            * af.exp(-m[0, 1] * (v1[:, 1] - v1_bulk)**2 / (2 * k * T_b)) \
            * af.exp(-m[0, 1] * (v2[:, 1] - v2_bulk_p)**2 / (2 * k * T_b)) \
            * af.exp(-m[0, 1] * (v3[:, 1] - v3_bulk_p)**2 / (2 * k * T_b))

    f = af.join(1, f_e, f_p)

    af.eval(f)
    return (f)

def initialize_E(q1, q2, params):

    E1 = 0 * q1**0
    E2 =   params.amplitude * -0.33905781194330314 * af.cos(params.k_q1 * q1) \
         - params.amplitude * -2.223221606811876e-14 * af.sin(params.k_q1 * q1)
    E3 =   params.amplitude * 0.2904170639549876 * af.cos(params.k_q1 * q1) \
         - params.amplitude * 0.00767462104112758 * af.sin(params.k_q1 * q1)

    return(E1, E2, E3)

def initialize_B(q1, q2, params):

    B1 = params.B0 * q1**0
    B2 =   params.amplitude * 0.4123231205757007 * af.cos(params.k_q1 * q1) \
         - params.amplitude * 0.010896135556966308 * af.sin(params.k_q1 * q1)
    B3 =   params.amplitude * 0.48138140773195076 * af.cos(params.k_q1 * q1)

    af.eval(B1, B2, B3)
    return(B1, B2, B3)
