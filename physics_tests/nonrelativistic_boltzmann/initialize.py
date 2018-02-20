"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_f(q1, q2, p1, p2, p3, params):

    m1 = params.mass[0, 0]
    m2 = params.mass[0, 1]

    k = params.boltzmann_constant

    rho_b = params.rho_background
    T_b   = params.temperature_background

    p1_bulk = params.p1_bulk_background
    p2_bulk = params.p2_bulk_background
    p3_bulk = params.p3_bulk_background

    pert_real = params.pert_real
    pert_imag = params.pert_imag

    k_q1 = params.k_q1
    k_q2 = params.k_q2

    # Calculating the perturbed density:
    rho_1 = rho_b + (  pert_real * af.cos(k_q1 * q1 + k_q2 * q2)
                     - pert_imag * af.sin(k_q1 * q1 + k_q2 * q2)
                    )
    
    rho_2 = rho_b + 0 * q1**0

    if(params.p_dim == 3):
        f = rho * (m / (2 * np.pi * k * T_b))**(3 / 2) \
                * af.exp(-m * (p1 - p1_bulk)**2 / (2 * k * T_b)) \
                * af.exp(-m * (p2 - p2_bulk)**2 / (2 * k * T_b)) \
                * af.exp(-m * (p3 - p3_bulk)**2 / (2 * k * T_b))

    elif(params.p_dim == 2):
        f = rho * (m / (2 * np.pi * k * T_b)) \
                * af.exp(-m * (p1 - p1_bulk)**2 / (2 * k * T_b)) \
                * af.exp(-m * (p2 - p2_bulk)**2 / (2 * k * T_b)) \

    else:
        
        f1 = rho_1 * (m1 / (2 * np.pi * k * T_b))**(1 / 2) \
                   * af.exp(-m1 * (p1[:, 0] - p1_bulk)**2 / (2 * k * T_b))
        f2 = rho_2 * (m2 / (2 * np.pi * k * T_b))**(1 / 2) \
                   * af.exp(-m2 * (p1[:, 1] - p1_bulk)**2 / (2 * k * T_b))
        f = af.join(1, f1, f2) 

    af.eval(f)
    return (f)

def initialize_A3_B3(q1, q2, params):

    A3 = af.sin(2 * np.pi * q1 + 4 * np.pi * q2)
    B3 = 1e-5 * q1**0

    return(A3, B3)
