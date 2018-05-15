"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np
import domain

N_q1 = domain.N_q1
N_q2 = domain.N_q2

N_p1 = domain.N_p1
N_p2 = domain.N_p2
N_p3 = domain.N_p3
N_g  = domain.N_ghost

def initialize_f(r, theta, rdot, thetadot, phidot, params):

	# The initialization is performed to setup particles
	# between r = 0.9 and 1.1 at theta = 0
    rho = af.select(r<2, 1 * r**0, 0)
    rho = af.select(r>1, rho, 0)
    rho = af.select(theta == 0, rho, 0)
    
    # Getting f at the right shape:
    f = rdot * rho

    f[:] = 0

    # Changing to velocities_expanded form:
    f        = af.moddims(f, N_p1, N_p2, N_p3, r.shape[2] * r.shape[3])
    rdot     = af.moddims(rdot, N_p1, N_p2, N_p3)
    thetadot = af.moddims(thetadot, N_p1, N_p2, N_p3)

    # Assigning the velocities such that rdot = r
    # Only looping over N_r since we're only initializing at theta = 0(at 0.5 * (N_q2 - 1))
    for i in range(r.shape[2] - 2 * N_g):
        rho_new                                        = rho * 0
        rho_new[:, :, N_g + i, 0.5 * (N_q2 - 1) + N_g] = rho[:, :,  N_g + i, 0.5 * (N_q2 - 1) + N_g]
        rho_new                                        =\
            af.moddims(rho_new, 1, 1, 1, r.shape[2] * r.shape[3])

        # We set thetadot = thetadot[16] for all particles and 
        # rdot proportional to r
        # At initialization:
        print('r        =', af.sum(r[:, :,  N_g + i, 0.5 * (N_q2 - 1) + N_g]))
        print('theta    =', af.sum(theta[:, :,  N_g + i, 0.5 * (N_q2 - 1) + N_g]))
        print('rdot     =', af.sum(rdot[i, 16, 0]))
        print('thetadot =', af.sum(thetadot[i, 16, 0]))
        f[i, 16, 0] = rho_new
    
    f = af.moddims(f, N_p1 * N_p2 * N_p3, 1, r.shape[2], r.shape[3])

    af.eval(f)
    return (f)
