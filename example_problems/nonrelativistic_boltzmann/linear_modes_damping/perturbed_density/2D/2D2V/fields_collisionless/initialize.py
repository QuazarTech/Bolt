"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_E(q1, q2, params):
    
    k_q1 = params.k_q1
    k_q2 = params.k_q2
    
    #E1 =   params.charge_electron * k_q1/(k_q1**2 + k_q2**2) \
    #     * (  params.pert_real * af.sin(k_q1 * q1 + k_q2 * q2) 
    #        + params.pert_imag * af.cos(k_q1 * q1 + k_q2 * q2)
    #       )

    #E2 =   params.charge_electron * k_q2/(k_q1**2 + k_q2**2) \
    #     * (  params.pert_real * af.sin(k_q1 * q1 + k_q2 * q2) 
    #        + params.pert_imag * af.cos(k_q1 * q1 + k_q2 * q2)
    #       )

    E1 = 2 * q1**0
    E2 = 3. * q1**0
    E3 = 0.  * q1**0

    return(E1, E2, E3)

def initialize_B(q1, q2, params):

    k_q1 = params.k_q1
    k_q2 = params.k_q2

    B1 = 0.  * q1**0
    B2 = 0.  * q1**0 
    B3 = 1.8 * q1**0

    return(B1, B2, B3)

def initialize_f(q1, q2, p1, p2, p3, params):

    m = params.mass_particle
    k = params.boltzmann_constant

    rho_b = params.rho_background
    T_b   = params.temperature_background

    p1_bulk = params.p1_bulk_background
    p2_bulk = params.p2_bulk_background

    pert_real = params.pert_real
    pert_imag = params.pert_imag

    k_q1 = params.k_q1
    k_q2 = params.k_q2

    # Calculating the perturbed density:
    rho = rho_b + (  pert_real * af.cos(k_q1 * q1 + k_q2 * q2)
                   - pert_imag * af.sin(k_q1 * q1 + k_q2 * q2)
                  )

    f = rho * (m / (2 * np.pi * k * T_b)) \
            * af.exp(-m * (p1 - p1_bulk)**2 / (2 * k * T_b)) \
            * af.exp(-m * (p2 - p2_bulk)**2 / (2 * k * T_b))

    af.eval(f)
    return (f)
