"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_f(q1, q2, v1, v2, v3, params):

    m = params.mass
    k = params.boltzmann_constant

    rho_b_1   = params.rho_background_species_1
    T_b_1     = params.temperature_background_species_1
    v1_bulk_1 = params.v1_bulk_background_species_1

    rho_b_2   = params.rho_background_species_2
    T_b_2     = params.temperature_background_species_2
    v1_bulk_2 = params.v1_bulk_background_species_2

    pert_real_species_1 = params.pert_real_species_1
    pert_imag_species_1 = params.pert_imag_species_1

    pert_real_species_2 = params.pert_real_species_2
    pert_imag_species_2 = params.pert_imag_species_2

    k_q1_species_1 = params.k_q1_species_1
    k_q2_species_1 = params.k_q2_species_1

    k_q1_species_2 = params.k_q1_species_2
    k_q2_species_2 = params.k_q2_species_2

    # Calculating the perturbed density:
    rho_species_1 = \
        rho_b_1 + (  pert_real_species_1 * af.cos(k_q1_species_1 * q1 + k_q2_species_1 * q2)
                   - pert_imag_species_1 * af.sin(k_q1_species_1 * q1 + k_q2_species_1 * q2)
                  )

    rho_species_2 = \
        rho_b_2 + (  pert_real_species_2 * af.cos(k_q1_species_2 * q1 + k_q2_species_2 * q2)
                   - pert_imag_species_2 * af.sin(k_q1_species_2 * q1 + k_q2_species_2 * q2)
                  )

    f_species_1 = rho_species_1 * np.sqrt(m[0] / (2 * np.pi * k * T_b_1)) \
                                * af.exp(-m[0] * (v1[:, 0] - v1_bulk_1)**2 / (2 * k * T_b_1))

    f_species_2 = rho_species_2 * np.sqrt(m[1] / (2 * np.pi * k * T_b_2)) \
                                * af.exp(-m[1] * (v1[:, 1] - v1_bulk_2)**2 / (2 * k * T_b_2))

    f = af.join(1, f_species_1, f_species_2)
    
    af.eval(f)
    return (f)
