"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_E(q1, q2, params):
    
    k_q1 = params.k_q1
    
    E1 =   params.charge_electron/params.k_q1 \
         * (params.pert_real * af.sin(k_q1 * q1) + params.pert_imag * af.cos(k_q1 * q1))

    E2 = E1
    E3 = 5 * q1**0

    return(E1, E2, E3)

def initialize_B(q1, q2, params):

    B1 = 2 * q1**0
    B2 = 1 * q1**0
    B3 =   params.pert_real * af.cos(params.k_q1 * q1) \
         - params.pert_imag * af.sin(params.k_q1 * q1)

    return(B1, B2, B3)

def initialize_f(q1, q2, p1, p2, p3, params):

    m = params.mass_particle
    k = params.boltzmann_constant

    rho_b = params.rho_background
    T_b   = params.temperature_background

    p1_bulk = params.p1_bulk_background

    pert_real = params.pert_real
    pert_imag = params.pert_imag

    k_q1 = params.k_q1

    # Calculating the perturbed density:
    rho = rho_b + (  pert_real * af.cos(k_q1 * q1)
                   - pert_imag * af.sin(k_q1 * q1)
                  )

    f = rho * np.sqrt(m / (2 * np.pi * k * T_b)) \
            * af.exp(-m * (p1 - p1_bulk)**2 / (2 * k * T_b))

    af.eval(f)
    return (f)
