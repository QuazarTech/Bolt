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

def initialize_f(r, theta, rdot, thetadot, phidot, params):

    rho = af.exp(-500 * (r * af.cos(theta) - 1)**2 - 500 * (r * af.sin(theta))**2)
    f   = rdot * rho

    f[:] = 0

    f   = af.moddims(f, N_p1, N_p2, N_p3, r.shape[2] * r.shape[3])
    rho = af.moddims(rho, 1, 1, 1, r.shape[2] * r.shape[3])

    f[24, 24, 0] = rho
    
    f = af.moddims(f, N_p1 * N_p2 * N_p3, 1, r.shape[2], r.shape[3])

    af.eval(f)
    return (f)
