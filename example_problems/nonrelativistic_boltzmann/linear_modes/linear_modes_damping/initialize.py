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

    v1_bulk = params.v1_bulk_background
    v2_bulk = params.v2_bulk_background
    v3_bulk = params.v3_bulk_background

    pert_real = params.pert_real
    pert_imag = params.pert_imag

    k_q1 = params.k_q1
    k_q2 = params.k_q2

    # Calculating the perturbed density using E1:
    E1 = 0.001 * af.sin(params.k_q1 * (q1-1/256)) / params.k_q1
    n  = n_b + (af.shift(E1, 0, 0, -1) - E1) * 128

    if(params.p_dim == 3):
        f = n * (m / (2 * np.pi * k * T_b))**(3 / 2) \
              * af.exp(-m * (v1 - v1_bulk)**2 / (2 * k * T_b)) \
              * af.exp(-m * (v2 - v2_bulk)**2 / (2 * k * T_b)) \
              * af.exp(-m * (v3 - v3_bulk)**2 / (2 * k * T_b))

    elif(params.p_dim == 2):
        f = n * (m / (2 * np.pi * k * T_b)) \
              * af.exp(-m * (v1 - v1_bulk)**2 / (2 * k * T_b)) \
              * af.exp(-m * (v2 - v2_bulk)**2 / (2 * k * T_b)) \

    else:
        f = n * (m / (2 * np.pi * k * T_b))**(1 / 2) \
              * af.exp(-m * (v1 - v1_bulk)**2 / (2 * k * T_b)) \

    af.eval(f)
    return (f)

def initialize_E(q1, q2, params):
    E1 = -0.01 * af.sin(params.k_q1 * q1) / params.k_q1
    E2 = 0 * q1**0
    E3 = 0 * q1**0

    return(E1, E2, E3)

def initialize_A3_B3(q1, q2, params):
    A3 = 0 * q1**0
    B3 = 0 * q1**0

    return(A3, B3)
