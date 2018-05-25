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
# L1_num  = pi
# k1_num  = 2 * pi / L1_num

# ('Eigenvalue   = ', -2.220382733940932e-15 - 3.0466686024511223*I)
# (delta_u2_e, ' = ', 8.413408858487514e-17 - 0.5389501869018833*I)
# (delta_u3_e, ' = ', 0.5389501869018835)
# (delta_u2_i, ' = ', -8.673617379884035e-17 - 0.09260702134169797*I)
# (delta_u3_i, ' = ', 0.09260702134169804 + 3.144186300207963e-18*I)
# (delta_B2, ' = ', 1.2793585635328952e-16 + 0.24600635942384688*I)
# (delta_B3, ' = ', -0.24600635942384713 + 7.502679033599691e-17*I)
# (delta_E2, ' = ', -0.37474992562996995 + 5.347285114698508e-16*I)
# (delta_E3, ' = ', 2.1076890233118206e-16 - 0.3747499256299707*I)

def initialize_f(q1, q2, v1, v2, v3, params):
    
    m = params.mass
    k = params.boltzmann_constant

    n_b = params.density_background
    T_b = params.temperature_background

    k = params.boltzmann_constant

    v1_bulk   = 0

    # Assigning separate bulk velocities
    v2_bulk_i =   params.amplitude * -8.673617379884035e-17 * af.cos(params.k_q1 * q1) \
    			- params.amplitude * - 0.09260702134169797 * af.sin(params.k_q1 * q1)

    v2_bulk_e =   params.amplitude * 8.413408858487514e-17 * af.cos(params.k_q1 * q1) \
                - params.amplitude * - 0.5389501869018833* af.sin(params.k_q1 * q1)
    
    v3_bulk_i =   params.amplitude * 0.09260702134169804 * af.cos(params.k_q1 * q1) \
    			- params.amplitude * 3.144186300207963e-18  * af.sin(params.k_q1 * q1)

    v3_bulk_e =   params.amplitude * 0.5389501869018835 * af.cos(params.k_q1 * q1) \
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
    
    E2 =   params.amplitude * -0.37474992562996995 * af.cos(params.k_q1 * q1) \
         - params.amplitude * 4.996003610813204e-16  * af.sin(params.k_q1 * q1)
    
    E3 =   params.amplitude * -4.7878367936959876e-15   * af.cos(params.k_q1 * q1) \
         - params.amplitude * - 0.3747499256299707 * af.sin(params.k_q1 * q1)

    af.eval(E1, E2, E3)
    return(E1, E2, E3)

def initialize_B(q1, q2, params):

    dt = params.dt
    B1 = params.B0 * q1**0

    omega = -2.220382733940932e-15 - 3.0466686024511223 * 1j

    B2 = (params.amplitude * (1.2793585635328952e-16 + 0.24600635942384688*1j) * \
          np.exp(  1j * params.k_q1 * np.array(q1)
                 + omega * dt / 2
                )).real

    B3 = (params.amplitude * (-0.24600635942384713 + 7.502679033599691e-17) * \
          np.exp(  1j * params.k_q1 * np.array(q1)
                 + omega * dt / 2
                )).real

    B2 = af.moddims(af.to_array(B2), 1, 1, q1.shape[2], q1.shape[3])
    B3 = af.moddims(af.to_array(B3), 1, 1, q1.shape[2], q1.shape[3])

    af.eval(B1, B2, B3)
    return(B1, B2, B3)
