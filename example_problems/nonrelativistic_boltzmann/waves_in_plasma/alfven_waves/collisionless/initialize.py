"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

# Problem Parameters:
# n0_num  = 1
# B10_num = 1
# c_num   = 8
# mu_num  = 1
# e_num   = 1
# mi_num  = 1
# me_num  = 1 / 10
# L1_num  = 200 * pi
# k1_num  = 2 * pi / L1_num


# ('Eigenvalue   = ', -2.066974648882578e-16 + 0.009427887494426246*I)
# (delta_u2_e, ' = ', -1.845078452002158e-12 - 0.34710064991731593*I)
# (delta_u3_e, ' = ', 0.3471006499240479 + 9.824606406194647e-15*I)
# (delta_u2_i, ' = ', -1.8449935589720523e-12 - 0.3507345786529984*I)
# (delta_u3_i, ' = ', 0.35073457865965896 - 9.728112412843437e-15*I)
# (delta_B2, ' = ', -1.9649296295956575e-12 - 0.36851085962829183*I)
# (delta_B3, ' = ', 0.3685108596354303)
# (delta_E2, ' = ', -0.34742789251171374 - 7.462997234086721e-15*I)
# (delta_E3, ' = ', -1.8452757767975503e-12 - 0.34742789250498896*I)

def initialize_f(q1, q2, v1, v2, v3, params):
    
    m = params.mass
    k = params.boltzmann_constant

    n_b = params.density_background
    T_b = params.temperature_background

    k = params.boltzmann_constant

    v1_bulk   = 0

    # Assigning separate bulk velocities
    v2_bulk_i =   params.amplitude * -1.8449935589720523e-12 * af.cos(params.k_q1 * q1) \
    			- params.amplitude * -0.3507345786529984 * af.sin(params.k_q1 * q1)

    v2_bulk_e =   params.amplitude * -1.845078452002158e-12 * af.cos(params.k_q1 * q1) \
                - params.amplitude * -0.34710064991731593 * af.sin(params.k_q1 * q1)
    
    v3_bulk_i =   params.amplitude * 0.35073457865965896 * af.cos(params.k_q1 * q1) \
    			- params.amplitude * -9.728112412843437e-15  * af.sin(params.k_q1 * q1)

    v3_bulk_e =   params.amplitude * 0.3471006499240479 * af.cos(params.k_q1 * q1) \
                - params.amplitude * 9.824606406194647e-15 * af.sin(params.k_q1 * q1)

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
    E2 =   params.amplitude * -0.34742789251171374 * af.cos(params.k_q1 * q1) \
         - params.amplitude * -7.462997234086721e-15 * af.sin(params.k_q1 * q1)
    E3 =   params.amplitude * -1.8452757767975503e-12 * af.cos(params.k_q1 * q1) \
         - params.amplitude * -0.34742789250498896 * af.sin(params.k_q1 * q1)

    return(E1, E2, E3)

def initialize_B(q1, q2, params):

    B1 = params.B0 * q1**0
    B2 =   params.amplitude * -1.9649296295956575e-12 * af.cos(params.k_q1 * q1) \
         - params.amplitude * -0.36851085962829183 * af.sin(params.k_q1 * q1)
    B3 =   params.amplitude * 0.3685108596354303 * af.cos(params.k_q1 * q1) \
         - params.amplitude * 0 * af.sin(params.k_q1 * q1)

    af.eval(B1, B2, B3)
    return(B1, B2, B3)
