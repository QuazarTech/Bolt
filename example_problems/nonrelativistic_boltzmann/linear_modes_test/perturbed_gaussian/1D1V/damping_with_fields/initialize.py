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

    # Calculating the perturbed density:
    rho = rho_b + 0.01 * af.exp(-(q1 - 0.5)**2 -
                                 (q2 - 0.5)**2)
    T   = T_b + 0.01 * af.exp(-(q1 - 0.5)**2 -
                               (q2 - 0.5)**2)

    p1_bulk = 0.01 * af.exp(-(q1 - 0.5)**2 -(q2 - 0.5)**2)
    p2_bulk = 0.01 * af.exp(-(q1 - 0.5)**2 -(q2 - 0.5)**2)
    p3_bulk = 0.01 * af.exp(-(q1 - 0.5)**2 -(q2 - 0.5)**2)

    # Depending on the dimensionality in velocity space, the
    # distribution function is assigned accordingly:
    if (params.p_dim == 3):

        f = rho_b * (m / (2 * np.pi * k * T))**(3 / 2) * \
            af.exp(-m * (p1 - p1_bulk)**2 / (2 * k * T)) * \
            af.exp(-m * (p2 - p2_bulk)**2 / (2 * k * T)) * \
            af.exp(-m * (p3 - p3_bulk)**2 / (2 * k * T))

    elif (params.p_dim == 2):

        f = rho_b * (m / (2 * np.pi * k * T)) * \
            af.exp(-m * (p1 - p1_bulk)**2 / (2 * k * T)) * \
            af.exp(-m * (p2 - p2_bulk)**2 / (2 * k * T))

    else:

        f = rho_b *\
            af.sqrt(m / (2 * np.pi * k * T)) * \
            af.exp(-m * (p1 - p1_bulk)**2 / (2 * k * T))

    af.eval(f)
    return (f)


# In case of user-defined fields the initial-conditions are provided
# using the functions below. It should be ensured that these satisfy
# maxwell's constraint equations.(add check - TODO)


def initialize_E(q1, q2, p1, p2, p3, params):
    # Declare initial conditions for E here
    E1 = 0 * q1[:, :, 0]
    E2 = 0 * q1[:, :, 0]
    E3 = 0 * q1[:, :, 0]

    af.eval(E1, E2, E3)
    return (E1, E2, E3)


def initialize_B(q1, q2, p1, p2, p3, params):
    # Declare initial conditions for B here
    B1 = 0 * q1[:, :, 0]
    B2 = 0 * q1[:, :, 0]
    B3 = 0 * q1[:, :, 0]

    af.eval(B1, B2, B3)
    return (B1, B2, B3)
