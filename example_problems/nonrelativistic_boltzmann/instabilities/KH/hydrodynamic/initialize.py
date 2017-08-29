"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_f(q1, q2, p1, p2, p3, params):

    m = params.mass_particle
    k = params.boltzmann_constant

    q2_minus = 0.5
    q2_plus = 1.5

    regulator = 20  # larger value makes the transition sharper

    rho = 1 + 0.5 * (af.tanh(( q2 - q2_minus)*regulator) -
                     af.tanh(( q2 - q2_plus)*regulator))

    p1_bulk = (af.tanh(( q2 - q2_minus)*regulator) -
               af.tanh(( q2 - q2_plus )*regulator) - 1) + \
               0.001 * af.to_array(np.random.choice([-1, 1],size = q1.shape) *
                                   np.random.rand(q1.shape[0], 
                                                  q1.shape[1]))

    p2_bulk = 0.5 * af.sin(2*np.pi*q1) *\
              (af.exp(-25 * (q2 - q2_minus)**2) +
               af.exp(-25 * (q2 - q2_plus)**2)) + \
               0.001 * af.to_array(np.random.choice([-1, 1],size = q1.shape) *
                                   np.random.rand(q1.shape[0], 
                                                  q1.shape[1]))

    T = (10 / rho)

    f = rho * (m / (2 * np.pi * k * T))**(3 / 2) * \
        af.exp(-m * (p1 - p1_bulk)**2 / (2 * k * T)) * \
        af.exp(-m * (p2 - p2_bulk)**2 / (2 * k * T)) * \
        af.exp(-m * (p3)**2 / (2 * k * T))

    af.eval(f)
    return (f)
