"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_f(q1, q2, v1, v2, v3, params):

    m = params.mass
    k = params.boltzmann_constant

    n_b       = params.density_background
    T_b       = params.temperature_background
    v1_bulk_b = params.v1_bulk_background
    v2_bulk_b = params.v2_bulk_background
    v3_bulk_b = params.v3_bulk_background

    pert_real_n = 1
    pert_imag_n = 0

    pert_real_v1 =  -np.sqrt(params.gamma * T_b) / n_b \
                   * params.k_q1 / np.sqrt(params.k_q1**2 + params.k_q2**2) * 1j
    pert_imag_v1 = 0

    pert_real_v2 =  -np.sqrt(params.gamma * T_b) / n_b \
                   * params.k_q2 / np.sqrt(params.k_q1**2 + params.k_q2**2) * 1j
    pert_imag_v2 = 0

    pert_real_T = T_b * (params.gamma - 1) / n_b
    pert_imag_T = 0

    k_q1 = params.k_q1
    k_q2 = params.k_q2
    amp  = params.amplitude

    # Introducing the perturbation amounts:
    # This is obtained from the Sage Worksheet(https://goo.gl/Sh8Nqt):
    # Plugging in the value from the Eigenvectors:
    # Calculating the perturbed density:
    n = n_b + amp * (  pert_real_n * af.cos(k_q1 * q1 + k_q2 * q2)
                     - pert_imag_n * af.sin(k_q1 * q1 + k_q2 * q2)
                    )

    # Calculating the perturbed bulk velocities:
    v1_bulk = v1_bulk_b + amp * (  pert_real_v1 * af.cos(k_q1 * q1 + k_q2 * q2)
                                 - pert_imag_v1 * af.sin(k_q1 * q1 + k_q2 * q2)
                                ) 
    v2_bulk = v2_bulk_b + amp * (  pert_real_v2 * af.cos(k_q1 * q1 + k_q2 * q2)
                                 - pert_imag_v2 * af.sin(k_q1 * q1 + k_q2 * q2)
                                ) 
    v3_bulk = v3_bulk_b

    # Calculating the perturbed temperature:
    T = T_b +  amp * (  pert_real_T * af.cos(k_q1 * q1 + k_q2 * q2)
                      - pert_imag_T * af.sin(k_q1 * q1 + k_q2 * q2)
                     )

    f = n * (m / (2 * np.pi * k * T))**(3 / 2) \
          * af.exp(-m * (v1 - v1_bulk)**2 / (2 * k * T))
          * af.exp(-m * (v2 - v2_bulk)**2 / (2 * k * T))
          * af.exp(-m * (v3 - v3_bulk)**2 / (2 * k * T))

    af.eval(f)
    return (f)
