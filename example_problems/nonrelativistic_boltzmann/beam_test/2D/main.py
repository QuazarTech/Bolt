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

# Time parameters:
dt      = 0.0025
t_final = 2.5

time_array = np.arange(dt, t_final + dt, dt)

n_nls = nls.compute_moments('density')

h5f = h5py.File('dump/0000.h5', 'w')
h5f.create_dataset('q1', data = nls.q1_center)
h5f.create_dataset('q2', data = nls.q2_center)
h5f.create_dataset('n', data = n_nls)
h5f.close()

def time_evolution():

    for time_index, t0 in enumerate(time_array):
        print('For Time =', t0)
        print('MIN(f) =', af.min(nls.f[3:-3, 3:-3]))
        print('MAX(f) =', af.max(nls.f[3:-3, 3:-3]))
        print('SUM(f) =', af.sum(nls.f[3:-3, 3:-3]))
        print()

        nls.strang_timestep(dt)
        n_nls = nls.compute_moments('density')
        
        h5f = h5py.File('dump/%04d'%(time_index+1) + '.h5', 'w')
        h5f.create_dataset('n', data = n_nls)
        h5f.close()

time_evolution()
