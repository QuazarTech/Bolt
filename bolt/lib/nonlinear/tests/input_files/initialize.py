"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_f(q1, q2, p1, p2, p3, params):

    m = params.mass
    k = params.boltzmann_constant

    q2_minus = 0.5
    q2_plus  = 1.5

    regulator = 20  # larger value makes the transition sharper

    rho = 1 + 0.5 * (  af.tanh(( q2 - q2_minus)*regulator) 
                     - af.tanh(( q2 - q2_plus )*regulator)
                    )

    p1_bulk = (  af.tanh(( q2 - q2_minus)*regulator)
               - af.tanh(( q2 - q2_plus )*regulator) - 1
              )

    p2_bulk = 0.5 * af.sin(2*np.pi*q1) *\
              (  af.exp(-25 * (q2 - q2_minus)**2)
               + af.exp(-25 * (q2 - q2_plus )**2)
              )
              
    T = (10 / rho)

    f = rho * (m / (2 * np.pi * k * T))**(3 / 2) \
            * af.exp(-m * (p1 - p1_bulk)**2 / (2 * k * T)) \
            * af.exp(-m * (p2 - p2_bulk)**2 / (2 * k * T)) \
            * af.exp(-m * (p3)**2 / (2 * k * T))

    af.eval(f)
    return (f)

def initialize_E(q1, q2, params):
    
    E1 = 0 * q1**0 
    E2 = 0 * q2**0 
    E3 = -6 * np.pi * af.cos(  2 * np.pi * q1
                             + 4 * np.pi * q2
                            )

    af.eval(E1, E2, E3)
    return(E1, E2, E3)

def initialize_A3_B3(q1, q2, params):

    A3 = af.sin(  2 * np.pi * q1
                + 4 * np.pi * q2
               )

    B3 = 0 * q1**0

    af.eval(A3, B3)
    return(A3, B3)
