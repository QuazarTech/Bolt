"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_f(q1, q2, v1, v2, v3, params):
    
    m       = params.mass
    k       = params.boltzmann_constant
    n       = params.n_background * q1**0
    v1_bulk = params.v1_bulk
    T       = params.T_background

    f = n * (m / (2 * np.pi * k * T))**(3 / 2) \
          * 0.5 * (  af.exp(-m * (v1 - v1_bulk)**2 / (2 * k * T))
                   + af.exp(-m * (v1 + v1_bulk)**2 / (2 * k * T))
                  ) \
          * af.exp(-m * (v2)**2 / (2 * k * T)) \
          * af.exp(-m * (v3)**2 / (2 * k * T))

    af.eval(f)
    return (f)

def initialize_E(q1, q2, params):

    E1 = 0 * q1**0
    E2 = 0 * q1**0
    E3 = 0 * q1**0

    af.eval(E1, E2, E3)
    return(E1, E2, E3)

def initialize_B(q1, q2, params):

    # Seeding with random fluctuation:
    B1 = params.B1 * af.randu(q1.shape[0], q1.shape[1], 
                              q1.shape[2], q1.shape[3],
                              dtype = af.Dtype.f64
                             )
    B2 = 0 * q1**0
    B3 = 0 * q1**0

    af.eval(B1, B2, B3)
    return(B1, B2, B3)
