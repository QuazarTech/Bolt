"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_f(q1, q2, v1, v2, v3, params):

    m = params.mass
    k = params.boltzmann_constant

    q2_minus = 0.5
    q2_plus  = 1.5

    regulator = 20  # larger value makes the transition sharper

    n = 1 + 0.5 * (  af.tanh(( q2 - q2_minus)*regulator) 
                   - af.tanh(( q2 - q2_plus )*regulator)
                  )

    v1_bulk = (  af.tanh(( q2 - q2_minus)*regulator)
               - af.tanh(( q2 - q2_plus )*regulator) - 1
              )

    v2_bulk = 0.01 * af.sin(2 * np.pi * q1) *\
              (  af.exp(-25 * (q2 - q2_minus)**2)
               + af.exp(-25 * (q2 - q2_plus )**2)
              )
              
    T = (10 / n)

    f = n * (m / (2 * np.pi * k * T)) \
          * af.exp(-m * (v1 - v1_bulk)**2 / (2 * k * T)) \
          * af.exp(-m * (v2 - v2_bulk)**2 / (2 * k * T)) 

    af.eval(f)
    return (f)
