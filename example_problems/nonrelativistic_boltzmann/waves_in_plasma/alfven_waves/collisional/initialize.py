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

    B3 = params.amplitude * np.sqrt(params.density_background) * af.cos(params.k_q1 * q1 / params.L_x)

    v1_bulk   = 0

    # Assigning separate bulk velocities to 
    v2_bulk_p = 0
    v2_bulk_e = -((af.shift(B3, 0, 0, -1, 0) - B3) / af.sum(q1[:, :, 1, 0] - q1[:, :, 0, 0])) / params.e_p
    
    v3_bulk   = params.amplitude * af.cos(params.k_q1 * q1 / params.L_x)

    n = n_b + 0 * q1**0

    f_e = n * (m[0, 0] / (2 * np.pi * k * T_b))**(3 / 2) \
            * af.exp(-m[0, 0] * (v1[:, 0] - v1_bulk)**2 / (2 * k * T_b)) \
            * af.exp(-m[0, 0] * (v2[:, 0] - v2_bulk_e)**2 / (2 * k * T_b)) \
            * af.exp(-m[0, 0] * (v3[:, 0] - v3_bulk)**2 / (2 * k * T_b))

    f_p = n * (m[0, 1] / (2 * np.pi * k * T_b))**(3 / 2) \
            * af.exp(-m[0, 1] * (v1[:, 1] - v1_bulk)**2 / (2 * k * T_b)) \
            * af.exp(-m[0, 1] * (v2[:, 1] - v2_bulk_p)**2 / (2 * k * T_b)) \
            * af.exp(-m[0, 1] * (v3[:, 1] - v3_bulk)**2 / (2 * k * T_b))

    f = af.join(1, f_e, f_p)

    af.eval(f)
    return (f)

def initialize_E(q1, q2, params):

    E1 = 0 * q1**0
    E2 = -params.amplitude * af.cos(params.k_q1 * q1 / params.L_x) * params.B0
    E3 = 0 * q1**0

    return(E1, E2, E3)

def initialize_B(q1, q2, params):

    B1 = params.B0 * q1**0
    B2 = 0 * q1**0
    B3 = params.amplitude * np.sqrt(params.density_background) * af.cos(params.k_q1 * q1 / params.L_x)

    af.eval(B1, B2, B3)
    return(B1, B2, B3)
