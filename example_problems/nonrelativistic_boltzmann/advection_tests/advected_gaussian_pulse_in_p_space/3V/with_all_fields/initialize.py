"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_f(q1, q2, p1, p2, p3, params):

    m = params.mass_particle
    k = params.boltzmann_constant

    rho_b = params.rho_background
    T_b   = params.temperature_background

    p1_bulk = params.p1_bulk_background
    p2_bulk = params.p2_bulk_background
    p3_bulk = params.p3_bulk_background

    rho = rho_b + 0 * q1
    f   = rho * (m / (2 * np.pi * k * T_b))**(3/2) \
              * af.exp(-m * (p1 - p1_bulk)**2 / (2 * k * T_b)) \
              * af.exp(-m * (p2 - p2_bulk)**2 / (2 * k * T_b)) \
              * af.exp(-m * (p3 - p3_bulk)**2 / (2 * k * T_b)) \

    af.eval(f)
    return (f)
