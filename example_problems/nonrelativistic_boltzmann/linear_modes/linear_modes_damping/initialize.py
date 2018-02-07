"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_f(q1, q2, p1, p2, p3, params):

    m = params.mass
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
    rho = rho_b + (  pert_real * af.cos(k_q1 * q1 + k_q2 * q2)
                   - pert_imag * af.sin(k_q1 * q1 + k_q2 * q2)
                  )

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
        f = rho * (m / (2 * np.pi * k * T_b))**(1 / 2) \
                * af.exp(-m * (p1 - p1_bulk)**2 / (2 * k * T_b)) \

    af.eval(f)
    return (f)

def initialize_A3_B3(q1, q2, params):
    A3 = af.sin(2 * np.pi * q1 + 4 * np.pi * q2)

    #dq1 = q1[1] - q1[0]
    #dq2 = q2[1] - q2[0]
    
    #B1 =  (af.shift(A3, 0, 0, -1) - A3) / dq2
    #B2 = -(af.shift(A3, 0, -1)    - A3) / dq1
    #B3 = 1e-5*q1**0.
    
    # B1 = -(1/np.sqrt(4 * np.pi)) * af.sin(2 * np.pi * q2)
    # B2 =  (1/np.sqrt(4 * np.pi)) * af.sin(4 * np.pi * q1)
    B3 = 1e-5 * q1**0

    return(A3, B3)
