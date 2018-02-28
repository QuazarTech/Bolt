"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np
from params import B0

def initialize_f(q1, q2, v1, v2, v3, params):
    
    f = 0 * q1**0 * v1**0
    
    af.eval(f)
    return (f)

def initialize_B(q1, q2, params):

    B1 = 1 * q1**0 * B0
    B2 = 0 * q1**0 * B0
    B3 = 0 * q1**0 * B0

    af.eval(B1, B2, B3)
    return(B1, B2, B3)
