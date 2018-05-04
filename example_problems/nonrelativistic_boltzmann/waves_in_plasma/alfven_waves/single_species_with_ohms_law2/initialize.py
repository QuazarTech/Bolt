"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

# Problem Parameters:
# n0_num  = 1
# B10_num = 1
# mu_num  = 1
# e_num   = 1
# mi_num  = 1
# L1_num  = 200 * pi
# k1_num  = 2 * pi / L1_num

# ('Eigenvalue   = ', -4.32617735111918e-19 - 0.009950124999218764*I)
# (delta_u2_i, ' = ', 5.079270337660091e-15 + 0.501248425808263*I)
# (delta_u3_i, ' = ', 0.5012484258083252)
# (delta_B2, ' = ', -5.051514762044462e-15 - 0.4987484492453844*I)
# (delta_B3, ' = ', -0.498748449245447 - 1.942890293094024e-16*I)

def initialize_f(q1, q2, v1, v2, v3, params):
    
    m = params.mass
    k = params.boltzmann_constant

    n_b = params.density_background
    T_b = params.temperature_background

    k = params.boltzmann_constant

    v1_bulk   = 0

    # Assigning separate bulk velocities
    v2_bulk =   params.amplitude * 5.079270337660091e-15 * af.cos(params.k_q1 * q1) \
              - params.amplitude * 0.501248425808263    * af.sin(params.k_q1 * q1)

    v3_bulk =   params.amplitude * 0.5012484258083252 * af.cos(params.k_q1 * q1) \
              - params.amplitude * 0                  * af.sin(params.k_q1 * q1)

    n = n_b + 0 * q1**0

    f = n * (m / (2 * np.pi * k * T_b)) \
          * af.exp(-m * (v2 - v2_bulk)**2 / (2 * k * T_b)) \
          * af.exp(-m * (v3 - v3_bulk)**2 / (2 * k * T_b))

    af.eval(f)
    return (f)

def initialize_E(q1, q2, params):

    v2_bulk =   params.amplitude * 5.079270337660091e-15 * af.cos(params.k_q1 * q1) \
              - params.amplitude * 0.501248425808263    * af.sin(params.k_q1 * q1)

    v3_bulk =   params.amplitude * 0.5012484258083252 * af.cos(params.k_q1 * q1) \
              - params.amplitude * 0                  * af.sin(params.k_q1 * q1)

    B1 = initialize_B(q1, q2, params)[0]
    B2 = initialize_B(q1, q2, params)[1]
    B3 = initialize_B(q1, q2, params)[2]

    dq1 = af.sum(q1[0, 0, 1, 0] - q1[0, 0, 0, 0])

    J2 = -(af.shift(B3, 0, 0, -1) - af.shift(B3, 0, 0, 1)) / (2 * dq1)
    J3 =  (af.shift(B2, 0, 0, -1) - af.shift(B2, 0, 0, 1)) / (2 * dq1)

    E1 = -(B3 * v2_bulk - B2 * v3_bulk) + (B3 * J2 - B2 * J3)
    E2 = -(B1 * v3_bulk - B3 * 0) + (B1 * J3 - B3 * 0)
    E3 = -(B2 * 0 - B1 * v2_bulk) + (B2 * 0 - B1 * J2)

    af.eval(E1, E2, E3)
    return(E1, E2, E3)

def initialize_B(q1, q2, params):

    B1 = params.B0 * q1**0
    B2 =   params.amplitude * -5.051514762044462e-15  * af.cos(params.k_q1 * q1) \
         - params.amplitude * -0.4987484492453844   * af.sin(params.k_q1 * q1)
    B3 =   params.amplitude * -0.498748449245447  * af.cos(params.k_q1 * q1)

    af.eval(B1, B2, B3)
    return(B1, B2, B3)
