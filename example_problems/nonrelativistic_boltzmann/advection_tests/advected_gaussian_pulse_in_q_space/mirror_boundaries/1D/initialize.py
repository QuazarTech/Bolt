"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_f(q1, q2, p1, p2, p3, params):

    rho = 0.01 * af.exp(-200 * (q2 - 0.5)**2)
    f   = p1 * rho

    f[:] = 0
    f[2] = rho
    
    af.eval(f)
    return (f)
