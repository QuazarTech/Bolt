"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np
from .solve_linear_modes import solve_linear_modes

def initialize_f(q1, q2, v1, v2, v3, params):

    m = params.mass
    k = params.boltzmann_constant

    n_b       = params.density_background
    T_b       = params.temperature_background
    v1_bulk_b = params.v1_bulk_background

    eigval, eigvecs = solve_linear_modes(params)

    pert_n      = eigvecs[0, 0]
    pert_real_n = pert_n.real
    pert_imag_n = pert_n.imag

    pert_v1      = eigvecs[1, 0]
    pert_real_v1 = pert_v1.real
    pert_imag_v1 = pert_v1.imag

    pert_T      = eigvecs[2, 0]
    pert_real_T = pert_T.real
    pert_imag_T = pert_T.imag

    k_q1 = params.k_q1
    amp  = params.amplitude

    # Introducing the perturbation amounts:
    # Plugging in the value from the Eigenvectors:
    # Calculating the perturbed density:
    n = n_b + amp * (  pert_real_n * af.cos(k_q1 * q1)
                     - pert_imag_n * af.sin(k_q1 * q1)
                    )

    # Calculating the perturbed bulk velocities:
    v1_bulk_s1 = v1_bulk_b + amp * (  pert_real_v1 * af.cos(k_q1 * q1)
                                    - pert_imag_v1 * af.sin(k_q1 * q1)
                                   ) 

    pert_v1      = eigvecs[1, 2]
    pert_real_v1 = pert_v1.real
    pert_imag_v1 = pert_v1.imag

    v1_bulk_s2 = v1_bulk_b + amp * (  pert_real_v1 * af.cos(k_q1 * q1)
                                    - pert_imag_v1 * af.sin(k_q1 * q1)
                                   ) 

    # Calculating the perturbed temperature:
    T = T_b +  amp * (  pert_real_T * af.cos(k_q1 * q1)
                      - pert_imag_T * af.sin(k_q1 * q1)
                     )

    f_1 = n * (m[:, 0] / (2 * np.pi * k * T))**(1 / 2) \
            * af.exp(-m[:, 0] * (v1[:, 0] - v1_bulk_s1)**2 / (2 * k * T))

    f_2 = n * (m[:, 1] / (2 * np.pi * k * T))**(1 / 2) \
            * af.exp(-m[:, 1] * (v1[:, 1] - v1_bulk_s2)**2 / (2 * k * T))

    f = af.join(1, f_1, f_2)
    af.eval(f)
    return (f)
