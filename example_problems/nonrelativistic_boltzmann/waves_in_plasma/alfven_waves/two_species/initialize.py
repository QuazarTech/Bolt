"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

# Problem Parameters:
# n0_num  = 1
# B10_num = 0.1
# c_num   = 32
# mu_num  = 1
# e_num   = 1
# mi_num  = 1
# me_num  = 1 / 10
# L1_num  = 200 * pi
# k1_num  = 2 * pi / L1_num

# ('Eigenvalue   = ', 1.0593552760005806e-16 + 0.009130300248103908*I)
# (delta_u2_e, ' = ', 0.3798666654207246 + 1.2132567899555274e-14*I)
# (delta_u3_e, ' = ', 1.1766573772189343e-13 + 0.3798666654200048*I)
# (delta_u2_i, ' = ', 0.4218512476401069)
# (delta_u3_i, ' = ', 1.1614291139613354e-13 + 0.42185124763945103*I)
# (delta_B2, ' = ', 0.4198492401273711 + 1.7588893259619723e-14*I)
# (delta_B3, ' = ', 1.21554560850428e-13 + 0.4198492401267336*I)
# (delta_E2, ' = ', -1.2009797461872578e-14 - 0.0383334962129536*I)
# (delta_E3, ' = ', 0.03833349621302522 + 1.2500963280623377e-15*I)

def initialize_f(q1, q2, v1, v2, v3, params):
    
    m = params.mass
    k = params.boltzmann_constant

    n_b = params.density_background
    T_b = params.temperature_background

    k = params.boltzmann_constant

    v1_bulk   = 0

    # Assigning separate bulk velocities
    v2_bulk_i =   params.amplitude * 0.4218512476401069 * af.cos(params.k_q1 * q1) \
    			- params.amplitude * 0 * af.sin(params.k_q1 * q1)

    v2_bulk_e =   params.amplitude * 0.3798666654207246 * af.cos(params.k_q1 * q1) \
                - params.amplitude * 0.3798666654207246 * af.sin(params.k_q1 * q1)
    
    v3_bulk_i =   params.amplitude * 1.1614291139613354e-13 * af.cos(params.k_q1 * q1) \
    			- params.amplitude * 0.42185124763945103  * af.sin(params.k_q1 * q1)

    v3_bulk_e =   params.amplitude * 1.1766573772189343e-13 * af.cos(params.k_q1 * q1) \
                - params.amplitude * 0.3798666654200048 * af.sin(params.k_q1 * q1)

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
    
    E2 =   params.amplitude * -1.2009797461872578e-14 * af.cos(params.k_q1 * q1) \
         - params.amplitude * -0.0383334962129536   * af.sin(params.k_q1 * q1)
    
    E3 =   params.amplitude * 0.03833349621302522   * af.cos(params.k_q1 * q1) \
         - params.amplitude * 1.2500963280623377e-15 * af.sin(params.k_q1 * q1)

    af.eval(E1, E2, E3)
    return(E1, E2, E3)

def initialize_B(q1, q2, params):

    B1 = params.B0 * q1**0
    
    B2 =   params.amplitude * 0.4198492401273711 * af.cos(params.k_q1 * q1) \
         - params.amplitude * 1.7588893259619723e-14 * af.sin(params.k_q1 * q1)

    B3 =   params.amplitude * 1.21554560850428e-13 * af.cos(params.k_q1 * q1) \
         - params.amplitude * 0.4198492401267336 * af.sin(params.k_q1 * q1)

    af.eval(B1, B2, B3)
    return(B1, B2, B3)
