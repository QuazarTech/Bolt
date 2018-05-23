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

# ('Eigenvalue   = ', 5.3944386867730924e-17 - 0.0898800439758432*I)
# (delta_u2_e, ' = ', -4.85722573273506e-15 - 0.333061857862197*I)
# (delta_u3_e, ' = ', -0.333061857862222 - 3.885780586188048e-16*I)
# (delta_u2_i, ' = ', -4.801714581503802e-15 - 0.3692429960259134*I)
# (delta_u3_i, ' = ', -0.3692429960259359 + 1.8041124150158794e-16*I)
# (delta_B2, ' = ', 5.6066262743570405e-15 + 0.37389325198333345*I)
# (delta_B3, ' = ', 0.37389325198336115)
# (delta_E2, ' = ', 0.336055419305355 + 4.996003610813204e-16*I)
# (delta_E3, ' = ', -4.7878367936959876e-15 - 0.33605541930533006*I)

def initialize_f(q1, q2, v1, v2, v3, params):
    
    m = params.mass
    k = params.boltzmann_constant

    n_b = params.density_background
    T_b = params.temperature_background

    k = params.boltzmann_constant

    v1_bulk   = 0

    # Assigning separate bulk velocities
    v2_bulk_i =   params.amplitude * -4.801714581503802e-15 * af.cos(params.k_q1 * q1) \
    			- params.amplitude * -0.3692429960259134 * af.sin(params.k_q1 * q1)

    v2_bulk_e =   params.amplitude * -4.85722573273506e-15 * af.cos(params.k_q1 * q1) \
                - params.amplitude * - 0.333061857862197* af.sin(params.k_q1 * q1)
    
    v3_bulk_i =   params.amplitude * -0.3692429960259359 * af.cos(params.k_q1 * q1) \
    			- params.amplitude * 1.8041124150158794e-16  * af.sin(params.k_q1 * q1)

    v3_bulk_e =   params.amplitude * -0.333061857862222 * af.cos(params.k_q1 * q1) \
                - params.amplitude * -3.885780586188048e-16 * af.sin(params.k_q1 * q1)

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
    
    E2 =   params.amplitude * 0.336055419305355 * af.cos(params.k_q1 * q1) \
         - params.amplitude * 4.996003610813204e-16  * af.sin(params.k_q1 * q1)
    
    E3 =   params.amplitude * -4.7878367936959876e-15   * af.cos(params.k_q1 * q1) \
         - params.amplitude * -0.33605541930533006 * af.sin(params.k_q1 * q1)

    af.eval(E1, E2, E3)
    return(E1, E2, E3)

def initialize_B(q1, q2, params):

    dt = params.dt
    B1 = params.B0 * q1**0

    omega = 5.3944386867730924e-17 - 0.0898800439758432 * 1j

    B2 = (params.amplitude * (5.6066262743570405e-15 + 0.37389325198333345*1j) * \
          np.exp(  1j * params.k_q1 * np.array(q1)
                 + omega * dt / 2
                )).real

    B3 = (params.amplitude * 0.37389325198336115 * \
          np.exp(  1j * params.k_q1 * np.array(q1)
                 + omega * dt / 2
                )).real

    B2 = af.moddims(af.to_array(B2), 1, 1, q1.shape[2], q1.shape[3])
    B3 = af.moddims(af.to_array(B3), 1, 1, q1.shape[2], q1.shape[3])

    af.eval(B1, B2, B3)
    return(B1, B2, B3)
