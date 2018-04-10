"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

# Problem Parameters:
# n0_num  = 1
# B10_num = 1
# c_num   = 5
# mu_num  = 1
# e_num   = 1
# mi_num  = 1
# me_num  = 1 / 100
# L1_num  = 200 * pi
# k1_num  = 2 * pi / L1_num


# ('Eigenvalue   = ', 4.336961335044048e-17 + 0.009804431676090668*I)
# (delta_u2_e, ' = ', 0.35265635158311565 - 1.6653345369377348e-15*I)
# (delta_u3_e, ' = ', 3.202958731574057e-13 - 0.352656351578324*I)
# (delta_u2_i, ' = ', 0.3491980868481956 - 4.864164626638967e-15*I)
# (delta_u3_i, ' = ', 3.202438314531264e-13 - 0.34919808684335674*I)
# (delta_B2, ' = ', 0.3596554979234367)
# (delta_B3, ' = ', 3.281333538218689e-13 - 0.3596554979185527*I)
# (delta_E2, ' = ', -3.2029934260435766e-13 + 0.3526217756272816*I)
# (delta_E3, ' = ', 0.3526217756320739 - 1.654926196081874e-15*I)

def initialize_f(q1, q2, v1, v2, v3, params):
    
    m = params.mass
    k = params.boltzmann_constant

    n_b = params.density_background
    T_b = params.temperature_background

    k = params.boltzmann_constant

    v1_bulk   = 0

    # Assigning separate bulk velocities
    v2_bulk_i =   params.amplitude * 0.3491980868481956 * af.cos(params.k_q1 * q1) \
    			- params.amplitude * -4.864164626638967e-15 * af.sin(params.k_q1 * q1)

    v2_bulk_e =   params.amplitude * 0.35265635158311565 * af.cos(params.k_q1 * q1) \
                - params.amplitude * -1.6653345369377348e-15 * af.sin(params.k_q1 * q1)
    
    v3_bulk_i =   params.amplitude * 3.202438314531264e-13 * af.cos(params.k_q1 * q1) \
    			- params.amplitude * -0.34919808684335674  * af.sin(params.k_q1 * q1)

    v3_bulk_e =   params.amplitude * 3.202958731574057e-13 * af.cos(params.k_q1 * q1) \
                - params.amplitude * -0.352656351578324 * af.sin(params.k_q1 * q1)

    n = n_b + 0 * q1**0

    f_e = n * (m[0, 0] / (2 * np.pi * k * T_b))**(3 / 2) \
            * af.exp(-m[0, 0] * (v1[:, 0] - v1_bulk)**2 / (2 * k * T_b)) \
            * af.exp(-m[0, 0] * (v2[:, 0] - v2_bulk_e)**2 / (2 * k * T_b)) \
            * af.exp(-m[0, 0] * (v3[:, 0] - v3_bulk_e)**2 / (2 * k * T_b))

    f_i = n * (m[0, 1] / (2 * np.pi * k * T_b))**(3 / 2) \
            * af.exp(-m[0, 1] * (v1[:, 1] - v1_bulk)**2 / (2 * k * T_b)) \
            * af.exp(-m[0, 1] * (v2[:, 1] - v2_bulk_i)**2 / (2 * k * T_b)) \
            * af.exp(-m[0, 1] * (v3[:, 1] - v3_bulk_i)**2 / (2 * k * T_b))

    f = af.join(1, f_e, f_i)

    af.eval(f)
    return (f)

def initialize_E(q1, q2, params):

    E1 = 0 * q1**0
    E2 =   params.amplitude * -3.2029934260435766e-13 * af.cos(params.k_q1 * q1) \
         - params.amplitude * 0.3526217756272816 * af.sin(params.k_q1 * q1)
    E3 =   params.amplitude * 0.3526217756320739 * af.cos(params.k_q1 * q1) \
         - params.amplitude * -1.654926196081874e-15 * af.sin(params.k_q1 * q1)

    return(E1, E2, E3)

def initialize_B(q1, q2, params):

    B1 = params.B0 * q1**0
    B2 =   params.amplitude * 0.3596554979234367 * af.cos(params.k_q1 * q1) \
         - params.amplitude * 0 * af.sin(params.k_q1 * q1)
    B3 =   params.amplitude * 3.281333538218689e-13 * af.cos(params.k_q1 * q1) \
         - params.amplitude * -0.3596554979185527 * af.sin(params.k_q1 * q1)

    af.eval(B1, B2, B3)
    return(B1, B2, B3)
