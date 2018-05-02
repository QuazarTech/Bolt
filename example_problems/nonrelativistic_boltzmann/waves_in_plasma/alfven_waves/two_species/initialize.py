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

# ('Eigenvalue   = ', -1.7252101918544516e-16 - 0.09073973512797716*I)
# (delta_u2_e, ' = ', -0.3338347605465225 + 6.696032617270475e-16*I)
# (delta_u3_e, ' = ', 1.744611399789875e-14 + 0.33383476054651806*I)
# (delta_u2_i, ' = ', -0.3704813476796817 + 2.3037127760972e-15*I)
# (delta_u3_i, ' = ', 1.7593565493356778e-14 + 0.37048134767967766*I)
# (delta_B2, ' = ', 0.3712419568409201)
# (delta_B3, ' = ', -1.8197249262996706e-14 - 0.37124195684091743*I)
# (delta_E2, ' = ', -1.739060284666749e-14 - 0.3368639683213682*I)
# (delta_E3, ' = ', -0.3368639683213722 + 8.465450562766819e-16*I)

def initialize_f(q1, q2, v1, v2, v3, params):
    
    m = params.mass
    k = params.boltzmann_constant

    n_b = params.density_background
    T_b = params.temperature_background

    k = params.boltzmann_constant

    v1_bulk   = 0

    # Assigning separate bulk velocities

    v2_bulk_i =   params.amplitude * -0.3704813476796817 * af.cos(params.k_q1 * q1) \
    			- params.amplitude * 2.3037127760972e-15 * af.sin(params.k_q1 * q1)

    v2_bulk_e =   params.amplitude * -0.3338347605465225   * af.cos(params.k_q1 * q1) \
                - params.amplitude * 6.696032617270475e-16 * af.sin(params.k_q1 * q1)
    
    v3_bulk_i =   params.amplitude * 1.7593565493356778e-14 * af.cos(params.k_q1 * q1) \
    			- params.amplitude * 0.37048134767967766  * af.sin(params.k_q1 * q1)

    v3_bulk_e =   params.amplitude * 1.744611399789875e-14 * af.cos(params.k_q1 * q1) \
                - params.amplitude * 0.33383476054651806 * af.sin(params.k_q1 * q1)

    n = n_b + 0 * q1**0

    f_e = n * (m[0, 0] / (2 * np.pi * k * T_b)) \
            * af.exp(-m[0, 0] * (v2[:, 0] - v2_bulk_e)**2 / (2 * k * T_b)) \
            * af.exp(-m[0, 0] * (v3[:, 0] - v3_bulk_e)**2 / (2 * k * T_b))

    f_i = n * (m[0, 1] / (2 * np.pi * k * T_b)) \
            * af.exp(-m[0, 1] * (v2[:, 1] - v2_bulk_i)**2 / (2 * k * T_b)) \
            * af.exp(-m[0, 1] * (v3[:, 1] - v3_bulk_i)**2 / (2 * k * T_b))

    f = af.join(1, f_e, f_i)

    af.eval(f)
    return (f)

def initialize_E(q1, q2, params):

    E1 = 0 * q1**0
    
    E2 =   params.amplitude * -1.739060284666749e-14 * af.cos(params.k_q1 * q1) \
         - params.amplitude * -0.3368639683213682    * af.sin(params.k_q1 * q1)
    
    E3 =   params.amplitude * -0.3368639683213722   * af.cos(params.k_q1 * q1) \
         - params.amplitude * 8.465450562766819e-16 * af.sin(params.k_q1 * q1)

    af.eval(E1, E2, E3)
    return(E1, E2, E3)

def initialize_B(q1, q2, params):

    B1 = params.B0 * q1**0
    
    B2 =   params.amplitude * 0.3712419568409201 * af.cos(params.k_q1 * q1) \
         - params.amplitude * 0 * af.sin(params.k_q1 * q1)

    B3 =   params.amplitude * -1.8197249262996706e-14 * af.cos(params.k_q1 * q1) \
         - params.amplitude * -0.37124195684091743 * af.sin(params.k_q1 * q1)

    af.eval(B1, B2, B3)
    return(B1, B2, B3)
