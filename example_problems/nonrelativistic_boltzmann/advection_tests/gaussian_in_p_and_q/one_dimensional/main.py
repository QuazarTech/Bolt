import arrayfire as af
import numpy as np
import h5py

from bolt.lib.physical_system import physical_system
from bolt.lib.nonlinear.nonlinear_solver import nonlinear_solver

import domain
import boundary_conditions
import params
import initialize

import bolt.src.nonrelativistic_boltzmann.advection_terms as advection_terms
import bolt.src.nonrelativistic_boltzmann.collision_operator as collision_operator
import bolt.src.nonrelativistic_boltzmann.moments as moments

# Defining the physical system to be solved:
system = physical_system(domain,
                         boundary_conditions,
                         params,
                         initialize,
                         advection_terms,
                         collision_operator.BGK,
                         moments
                        )

# Declaring a linear system object which will evolve the defined physical system:
nls = nonlinear_solver(system)
N_g = nls.N_ghost

# Time parameters:
dt      = 0.001 / 16
t_final = 1.0

time_array = np.arange(dt, t_final + dt, dt)

nls.dump_distribution_function('dump_f/0000')
nls.dump_moments('dump_moments/0000')

# n_nls = nls.compute_moments('density')
# h5f = h5py.File('dump/0000.h5', 'w')
# h5f.create_dataset('q1', data = nls.q1_center[:, :, N_g:-N_g, N_g:-N_g])
# h5f.create_dataset('q2', data = nls.q2_center[:, :, N_g:-N_g, N_g:-N_g])
# h5f.create_dataset('n', data = n_nls[:, :, N_g:-N_g, N_g:-N_g])
# h5f.close()

init_sum = af.sum(nls.f[:, :, N_g:-N_g, N_g:-N_g])

for time_index, t0 in enumerate(time_array):

    # Used to debug:    
    print('For Time =', t0)
    print('MIN(f) =', af.min(nls.f[:, :, N_g:-N_g, N_g:-N_g]))
    print('MAX(f) =', af.max(nls.f[:, :, N_g:-N_g, N_g:-N_g]))
    print('d(SUM(f)) =', af.sum(nls.f[:, :, N_g:-N_g, N_g:-N_g]) - init_sum)
    print()

    nls.strang_timestep(dt)
    n_nls = nls.compute_moments('density')

    # n_nls = nls.compute_moments('density')
    # h5f = h5py.File('dump/%04d'%(time_index+1) + '.h5', 'w')
    # h5f.create_dataset('q1', data = nls.q1_center[:, :, N_g:-N_g, N_g:-N_g])
    # h5f.create_dataset('q2', data = nls.q2_center[:, :, N_g:-N_g, N_g:-N_g])
    # h5f.create_dataset('n', data = n_nls[:, :, N_g:-N_g, N_g:-N_g])
    # h5f.close()

    # nls.dump_moments('dump_moments/%04d'%(time_index+1))
    
    if((time_index + 1) % 16 == 0):
        nls.dump_distribution_function('dump_f/%04d'%((time_index+1) / 16))
