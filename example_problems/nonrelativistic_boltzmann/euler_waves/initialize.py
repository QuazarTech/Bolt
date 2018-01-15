"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_f(q1, q2, v1, v2, v3, params):

    m = params.mass
    k = params.boltzmann_constant

    rho_b     = params.rho_background
    T_b       = params.temperature_background
    v1_bulk_b = params.v1_bulk_background
    v2_bulk_b = params.v2_bulk_background
    v3_bulk_b = params.v3_bulk_background

    pert_real_rho = params.pert_rho.real
    pert_imag_rho = params.pert_rho.imag

    pert_real_T = params.pert_T.real
    pert_imag_T = params.pert_T.imag

    pert_real_v1 = params.pert_v1.real
    pert_imag_v1 = params.pert_v1.imag

    k_q1 = params.k_q1
    k_q2 = params.k_q2

    # Calculating the perturbed density:
    rho = rho_b + (  pert_real_rho * af.cos(k_q1 * q1 + k_q2 * q2)
                   - pert_imag_rho * af.sin(k_q1 * q1 + k_q2 * q2)
                  )

    # Calculating the perturbed bulk velocities:
    v1_bulk = v1_bulk_b + (  pert_real_v1 * af.cos(k_q1 * q1 + k_q2 * q2)
                           - pert_imag_v1 * af.sin(k_q1 * q1 + k_q2 * q2)
                          )
    v2_bulk = v2_bulk_b
    v3_bulk = v3_bulk_b

    # Calculating the perturbed temperature:
    T = T_b + (  pert_real_T * af.cos(k_q1 * q1 + k_q2 * q2)
               - pert_imag_T * af.sin(k_q1 * q1 + k_q2 * q2)
              )

    f = rho * (m / (2 * np.pi * k * T))**(3 / 2) \
            * af.exp(-m * (v1 - v1_bulk)**2 / (2 * k * T)) \
            * af.exp(-m * (v2 - v2_bulk)**2 / (2 * k * T)) \
            * af.exp(-m * (v3 - v3_bulk)**2 / (2 * k * T))

    af.eval(f)
    return (f)
