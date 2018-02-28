"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np
import domain

N_p1 = domain.N_p1
N_p2 = domain.N_p2
N_p3 = domain.N_p3

def initialize_f(q1, q2, p1, p2, p3, params):
    rho = af.exp(-200 * (q1 - 0.5)**2 - 200 * (q2 - 0.5)**2)
    f   = p1 * rho

    f[:] = 0

    f   = af.moddims(f, N_p1, N_p2, N_p3, q1.shape[2] * q1.shape[3])
    rho = af.moddims(rho, 1, 1, 1, q1.shape[2] * q1.shape[3])

    f[2, 2, 0] = rho
    
    f = af.moddims(f, N_p1 * N_p2 * N_p3, 1, q1.shape[2], q1.shape[3])

    af.eval(f)
    return (f)

