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
n_nls = nls.compute_moments('density')
p1_bulk_nls = nls.compute_moments('mom_p1_bulk') / n_nls
p2_bulk_nls = nls.compute_moments('mom_p2_bulk') / n_nls
p3_bulk_nls = nls.compute_moments('mom_p3_bulk') / n_nls
T_nls = (  nls.compute_moments('energy')
         - n_nls * p1_bulk_nls**2
         - n_nls * p2_bulk_nls**2
         - n_nls * p3_bulk_nls**2
        ) / (params.p_dim * n_nls)

h5f = h5py.File('dump/0000.h5', 'w')
h5f.create_dataset('q1', data = nls.q1_center)
h5f.create_dataset('q2', data = nls.q2_center)
h5f.create_dataset('n', data = n_nls)
h5f.create_dataset('p1', data = p1_bulk_nls)
h5f.create_dataset('T', data = T_nls)
h5f.close()

# Time parameters:
dt      = 0.00005
t_final = 0.2

time_array  = np.arange(dt, t_final + dt, dt)

for time_index, t0 in enumerate(time_array):
    
    if(time_index%100 == 0):
        print('Computing for Time =', t0)

    nls.strang_timestep(dt)
    
    n_nls = nls.compute_moments('density')
    p1_bulk_nls = nls.compute_moments('mom_p1_bulk') / n_nls
    p2_bulk_nls = nls.compute_moments('mom_p2_bulk') / n_nls
    p3_bulk_nls = nls.compute_moments('mom_p3_bulk') / n_nls
    T_nls = (  nls.compute_moments('energy')
             - n_nls * p1_bulk_nls**2
             - n_nls * p2_bulk_nls**2
             - n_nls * p3_bulk_nls**2
            ) / (params.p_dim * n_nls)
    
    h5f = h5py.File('dump/%04d'%(time_index+1) + '.h5', 'w')
    h5f.create_dataset('n', data = n_nls)
    h5f.create_dataset('p1', data = p1_bulk_nls)
    h5f.create_dataset('T', data = T_nls)
    h5f.close()
