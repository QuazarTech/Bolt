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
# me_num  = 1 / 10
# L1_num  = 2 / 2.1 * pi
# k1_num  = 2 * pi / L1_num

# ('Eigenvalue   = ', -7.403321437792102e-18 - 0.8375236525870918*I)
# (delta_u2_e, ' = ', 0.09618006271785962 + 6.982261990806649e-17*I)
# (delta_u3_e, ' = ', -7.502679033599691e-17 - 0.09618006271785998*I)
# (delta_u2_i, ' = ', 0.6415418128296125)
# (delta_u3_i, ' = ', 9.71445146547012e-17 - 0.6415418128296122*I)
# (delta_B2, ' = ', -0.2613589207807856 - 1.1362438767648086e-16*I)
# (delta_B3, ' = ', 1.1275702593849246e-17 + 0.26135892078078615*I)
# (delta_E2, ' = ', 7.37257477290143e-17 + 0.10423537046121174*I)
# (delta_E3, ' = ', 0.10423537046121167 - 1.6479873021779667e-17*I)

def initialize_f(q1, q2, v1, v2, v3, params):
    
    m = params.mass
    k = params.boltzmann_constant

    n_b = params.density_background
    T_b = params.temperature_background

    k = params.boltzmann_constant

    v1_bulk   = 0

    # Assigning separate bulk velocities
    v2_bulk_i =   params.amplitude * 0.6415418128296125 * af.cos(params.k_q1 * q1) \
                - params.amplitude * 0 * af.sin(params.k_q1 * q1)

    v2_bulk_e =   params.amplitude * 0.09618006271785962 * af.cos(params.k_q1 * q1) \
                - params.amplitude * 0 * af.sin(params.k_q1 * q1)
    
    v3_bulk_i =   params.amplitude * 0 * af.cos(params.k_q1 * q1) \
                - params.amplitude * - 0.6415418128296122  * af.sin(params.k_q1 * q1)

    v3_bulk_e =   params.amplitude * 0 * af.cos(params.k_q1 * q1) \
                - params.amplitude * - 0.09618006271785998 * af.sin(params.k_q1 * q1)

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
    
    E2 =   params.amplitude * 0 * af.cos(params.k_q1 * q1) \
         - params.amplitude * 0.10423537046121174  * af.sin(params.k_q1 * q1)
    
    E3 =   params.amplitude * 0.10423537046121167   * af.cos(params.k_q1 * q1) \
         - params.amplitude * 0 * af.sin(params.k_q1 * q1)

    af.eval(E1, E2, E3)
    return(E1, E2, E3)

def initialize_B(q1, q2, params):

    dt = params.dt
    B1 = params.B0 * q1**0

    omega = -7.403321437792102e-18 - 0.8375236525870918 * 1j

    B2 = (params.amplitude * (-0.2613589207807856 - 1.1362438767648086e-16*1j) * \
          np.exp(  1j * params.k_q1 * np.array(q1)
                 + omega * dt / 2
                )).real

    B3 = (params.amplitude * (1.1275702593849246e-17 + 0.26135892078078615*1j) * \
          np.exp(  1j * params.k_q1 * np.array(q1)
                 + omega * dt / 2
                )).real

    B2 = af.moddims(af.to_array(B2), 1, 1, q1.shape[2], q1.shape[3])
    B3 = af.moddims(af.to_array(B3), 1, 1, q1.shape[2], q1.shape[3])

    af.eval(B1, B2, B3)
    return(B1, B2, B3)
