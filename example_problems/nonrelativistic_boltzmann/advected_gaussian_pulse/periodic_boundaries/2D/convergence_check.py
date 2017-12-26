import arrayfire as af
import numpy as np
import h5py

from bolt.lib.physical_system import physical_system

from bolt.lib.nonlinear_solver.nonlinear_solver \
    import nonlinear_solver

import domain
import boundary_conditions
import params
import initialize

import bolt.src.nonrelativistic_boltzmann.advection_terms as advection_terms

import bolt.src.nonrelativistic_boltzmann.collision_operator \
    as collision_operator

import bolt.src.nonrelativistic_boltzmann.moment_defs as moment_defs

N     = 2**np.arange(5, 10)
error = np.zeros(N.size)

for i in range(N.size):
    
    domain.N_q1 = int(N[i])
    domain.N_q2 = int(N[i])

    # Defining the physical system to be solved:
    system = physical_system(domain,
                             boundary_conditions,
                             params,
                             initialize,
                             advection_terms,
                             collision_operator.BGK,
                             moment_defs
                            )

    # Declaring a linear system object which will evolve the defined physical system:
    nls = nonlinear_solver(system)
    N_g = nls.N_ghost_q

    # Time parameters:
    dt      = 0.01 * 32/nls.N_q1
    t_final = 0.1

    time_array = np.arange(dt, t_final + dt, dt)
    from initialize import initialize_f
    f_initial = 0.01 * af.exp(-500 * (nls.q1_center - 0.6)**2 - 500 * (nls.q2_center - 0.6)**2)
    add = lambda a, b:a+b
    #f_initial  = af.broadcast(intialize_f, af.broadcast(nls.q1_center, nls.q2_center, 

    for time_index, t0 in enumerate(time_array):
        nls.strang_timestep(dt)

    error[i] = af.mean(af.abs(  nls.f[10, N_g:-N_g, N_g:-N_g] 
                              - f_initial[:, N_g:-N_g, N_g:-N_g]
                             )
                      )

print(error)
print(np.polyfit(np.log10(N), np.log10(error), 1))
