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
# L1_num  = 2 / 1.9 * pi
# k1_num  = 2 * pi / L1_num

# ('Eigenvalue   = ', -5.3757945175912085e-17 - 0.8125616518681449*I)
# (delta_u2_e, ' = ', 0.10931992932837933 - 1.0001765041178778e-17*I)
# (delta_u3_e, ' = ', -2.0911549401814167e-17 - 0.10931992932838053*I)
# (delta_u2_i, ' = ', 0.6306225419834888)
# (delta_u3_i, ' = ', -9.546400128734867e-17 - 0.6306225419834885*I)
# (delta_B2, ' = ', -0.2763918403673667 + 1.6263032587282567e-19*I)
# (delta_B3, ' = ', -3.4423418976414766e-17 + 0.27639184036736697*I)
# (delta_E2, ' = ', 3.2051726724102725e-17 + 0.1182028475640979*I)
# (delta_E3, ' = ', 0.1182028475640971 + 1.2368036282628392e-16*I)

def initialize_f(q1, q2, v1, v2, v3, params):
    
    m = params.mass
    k = params.boltzmann_constant

    n_b = params.density_background
    T_b = params.temperature_background

    k = params.boltzmann_constant

    v1_bulk   = 0

    # Assigning separate bulk velocities
    v2_bulk_i =   params.amplitude * 0.6306225419834888 * af.cos(params.k_q1 * q1) \
                - params.amplitude * 0 * af.sin(params.k_q1 * q1)

    v2_bulk_e =   params.amplitude * 0.10931992932837933 * af.cos(params.k_q1 * q1) \
                - params.amplitude * 0 * af.sin(params.k_q1 * q1)
    
    v3_bulk_i =   params.amplitude * 0 * af.cos(params.k_q1 * q1) \
                - params.amplitude * -0.6306225419834885  * af.sin(params.k_q1 * q1)

    v3_bulk_e =   params.amplitude * 0 * af.cos(params.k_q1 * q1) \
                - params.amplitude * -0.10931992932838053 * af.sin(params.k_q1 * q1)

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
         - params.amplitude * 0.1182028475640979  * af.sin(params.k_q1 * q1)
    
    E3 =   params.amplitude * 0.1182028475640971  * af.cos(params.k_q1 * q1) \
         - params.amplitude * 0 * af.sin(params.k_q1 * q1)

    af.eval(E1, E2, E3)
    return(E1, E2, E3)

def initialize_B(q1, q2, params):

    dt = params.dt
    B1 = params.B0 * q1**0

    omega = -5.3757945175912085e-17 - 0.8125616518681449 * 1j

    B2 = (params.amplitude * (-0.2763918403673667 + 1.6263032587282567e-19*1j) * \
          np.exp(  1j * params.k_q1 * np.array(q1)
                 + omega * dt / 2
                )).real

    B3 = (params.amplitude * (-3.4423418976414766e-17 + 0.27639184036736697*1j) * \
          np.exp(  1j * params.k_q1 * np.array(q1)
                 + omega * dt / 2
                )).real

    B2 = af.moddims(af.to_array(B2), 1, 1, q1.shape[2], q1.shape[3])
    B3 = af.moddims(af.to_array(B3), 1, 1, q1.shape[2], q1.shape[3])

    af.eval(B1, B2, B3)
    return(B1, B2, B3)
