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

N_g = system.N_ghost

# Declaring the solver object which will evolve the defined physical system:
nls = nonlinear_solver(system)

# Timestep as set by the CFL condition:
dt = params.N_cfl * min(nls.dq1, nls.dq2) \
                  / max(domain.p1_end, domain.p2_end, domain.p3_end)

time_array = np.arange(0, params.t_final + dt, dt)
n_data     = np.zeros([time_array.size, 2])
E_data     = np.zeros([time_array.size])

# Storing data at time t = 0:
n            = nls.compute_moments('density')
n_data[0, 0] = af.max(n[:, 0, N_g:-N_g, N_g:-N_g])
n_data[0, 1] = af.max(n[:, 1, N_g:-N_g, N_g:-N_g])
E_data[0]    = af.max(nls.fields_solver.cell_centered_EM_fields[:, :, N_g:-N_g, N_g:-N_g])

for time_index, t0 in enumerate(time_array[1:]):

    print('Computing For Time =', t0)
    nls.strang_timestep(dt)

    n                         = nls.compute_moments('density')
    n_data[time_index + 1, 0] = af.max(n[:, 0, N_g:-N_g, N_g:-N_g])
    n_data[time_index + 1, 1] = af.max(n[:, 1, N_g:-N_g, N_g:-N_g])
    E_data[time_index + 1] = \
        af.max(nls.fields_solver.cell_centered_EM_fields[:, :, N_g:-N_g, N_g:-N_g])

h5f = h5py.File('data_nls.h5', 'w')
h5f.create_dataset('E', data = E_data)
h5f.create_dataset('n', data = n_data)
h5f.create_dataset('time', data = time_array)
h5f.close()
