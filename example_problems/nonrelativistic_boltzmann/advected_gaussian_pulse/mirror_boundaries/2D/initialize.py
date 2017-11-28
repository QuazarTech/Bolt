"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np
import domain

N_q1 = domain.N_q1
N_q2 = domain.N_q2
N_g  = domain.N_ghost

N_p1 = domain.N_p1
N_p2 = domain.N_p2
N_p3 = domain.N_p3

def initialize_f(q1, q2, p1, p2, p3, params):

    m = params.mass_particle
    k = params.boltzmann_constant

    rho_b = params.rho_background

    # Calculating the perturbed density:
    rho = 0.01 * af.exp(-500 * (q1 - 0.5)**2 - 500 * (q2 - 0.5)**2)
    f   = p1 * rho

    f[:] = 0

    f   = af.moddims(f, N_p1, N_p2, N_p3, (N_q1 + 2 * N_g) * (N_q2 + 2 * N_g))
    rho = af.moddims(rho, 1, 1, 1, (N_q1 + 2 * N_g) * (N_q2 + 2 * N_g))

    f[2, 2, 0] = rho
    
    f = af.moddims(f, N_p1 * N_p2 * N_p3, (N_q1 + 2 * N_g), (N_q2 + 2 * N_g))

    af.eval(f)
    return (f)
