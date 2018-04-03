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

    v1_bulk   = 0

    # Assigning separate bulk velocities
    v2_bulk_p =   params.amplitude * 1.9229717500626917e-05 * af.cos(params.k_q1 * q1 / params.L_x) \
    			- params.amplitude * 0.00655786394798061 * af.sin(params.k_q1 * q1 / params.L_x)

    v2_bulk_e =   params.amplitude * 1.9229717500755067e-05 * af.cos(params.k_q1 * q1 / params.L_x) \
                - params.amplitude * 0.012880830658175434 * af.sin(params.k_q1 * q1 / params.L_x)
    
    v3_bulk_p =   params.amplitude * -0.447043712296218 * af.cos(params.k_q1 * q1 / params.L_x) \
    			+ params.amplitude * 1.359813681541238e-07* af.sin(params.k_q1 * q1 / params.L_x)

    v3_bulk_e =   params.amplitude * -0.4471811611874473 * af.cos(params.k_q1 * q1 / params.L_x) \
                + params.amplitude * 1.3596110752436136e-07 * af.sin(params.k_q1 * q1 / params.L_x)

    n = n_b + 0 * q1**0

    f_e = n * (m[0, 0] / (2 * np.pi * k * T_b))**(3 / 2) \
            * af.exp(-m[0, 0] * (v1[:, 0] - v1_bulk)**2 / (2 * k * T_b)) \
            * af.exp(-m[0, 0] * (v2[:, 0] - v2_bulk_e)**2 / (2 * k * T_b)) \
            * af.exp(-m[0, 0] * (v3[:, 0] - v3_bulk_e)**2 / (2 * k * T_b))

    f_p = n * (m[0, 1] / (2 * np.pi * k * T_b))**(3 / 2) \
            * af.exp(-m[0, 1] * (v1[:, 1] - v1_bulk)**2 / (2 * k * T_b)) \
            * af.exp(-m[0, 1] * (v2[:, 1] - v2_bulk_p)**2 / (2 * k * T_b)) \
            * af.exp(-m[0, 1] * (v3[:, 1] - v3_bulk_p)**2 / (2 * k * T_b))

    f = af.join(1, f_e, f_p)

    af.eval(f)
    return (f)

def initialize_E(q1, q2, params):
    E1 = 0 * q1**0
    E2 =   params.amplitude * 0.44709008792832566 * af.cos(params.k_q1 * q1 / params.L_x) \
         - params.amplitude * 1.5156847586912658e-10 * af.sin(params.k_q1 * q1 / params.L_x)
    E3 =   params.amplitude * 1.9228157352076084e-05 * af.cos(params.k_q1 * q1 / params.L_x) \
         - params.amplitude * 0.00971886132961468 * af.sin(params.k_q1 * q1 / params.L_x)

    return(E1, E2, E3)

def initialize_B(q1, q2, params):

    B1 = params.B0 * q1**0
    B2 =   params.amplitude * -2.7194247564653833e-05 * af.cos(params.k_q1 * q1 / params.L_x) \
         + params.amplitude * 0.013744889122931292 * af.sin(params.k_q1 * q1 / params.L_x)
    B3 =   params.amplitude * 0.6322966710194697 * af.cos(params.k_q1 * q1 / params.L_x)

    af.eval(B1, B2, B3)
    return(B1, B2, B3)
