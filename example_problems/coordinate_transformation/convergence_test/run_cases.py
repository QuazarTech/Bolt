import arrayfire as af
import numpy as np
import h5py

from bolt.lib.physical_system import physical_system
from bolt.lib.nonlinear.nonlinear_solver import nonlinear_solver

import domain
import boundary_conditions
import params
import initialize

import bolt.src.coordinate_transformation.advection_terms as advection_terms
import bolt.src.coordinate_transformation.source_term as source_term
import bolt.src.coordinate_transformation.moments as moments

N     = np.array([32, 48, 64, 96, 112]) #, 128, 144, 160])
error = np.zeros(N.size)

for i in range(N.size):
    af.device_gc()

    domain.N_q1 = int(N[i])
    domain.N_q2 = int(N[i])
    domain.N_p1 = int(N[i])
    domain.N_p2 = int(N[i])

    # Defining the physical system to be solved:
    system = physical_system(domain,
                             boundary_conditions,
                             params,
                             initialize,
                             advection_terms,
                             source_term.source_term,
                             moments
                            )

    # Declaring a linear system object which will evolve the defined physical system:
    nls = nonlinear_solver(system)
    N_g = nls.N_ghost

    # DEBUGGING:
    # h5f = h5py.File('data.h5', 'w')
    # h5f.create_dataset('q1', data = nls.q1_center)
    # h5f.create_dataset('q2', data = nls.q2_center)
    # h5f.create_dataset('p1', data = nls.p1_center)
    # h5f.create_dataset('p2', data = nls.p2_center)
    # h5f.create_dataset('p3', data = nls.p3_center)
    # h5f.close()

    # Time parameters:
    dt      = 0.001 * 32/nls.N_q1
    t_final = params.t_final

    time_array  = np.arange(dt, t_final + dt, dt)

    for time_index, t0 in enumerate(time_array):
        nls.strang_timestep(dt)

    nls.dump_distribution_function('dump/%04d'%int(N[i]))
