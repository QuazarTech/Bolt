import arrayfire as af
import numpy as np
import h5py

from bolt.lib.physical_system import physical_system
from bolt.lib.linear.linear_solver import linear_solver
from bolt.lib.utils.fft_funcs import ifft2

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

# Declaring the solver object which will evolve the defined physical system:
ls  = linear_solver(system)

# Timestep as set by the CFL condition:
dt = params.N_cfl * min(ls.dq1, ls.dq2) \
                  / max(domain.p1_end, domain.p2_end, domain.p3_end)

time_array = np.arange(0, params.t_final + dt, dt)
n_data     = np.zeros([time_array.size, 2])
E_data     = np.zeros([time_array.size])

# Storing data at time t = 0:

n            = ls.compute_moments('density')
n_data[0, 0] = af.max(n[:, 0]) 
n_data[0, 1] = af.max(n[:, 1]) 

E1_ls      = af.real(0.5 * (ls.N_q1 * ls.N_q2) 
                         * ifft2(ls.fields_solver.E1_hat)
                    )
E_data[0]  = af.max(E1_ls)

for time_index, t0 in enumerate(time_array[1:]):

    print('Computing For Time =', t0)
    ls.RK5_timestep(dt)
    
    n                         = ls.compute_moments('density')
    n_data[time_index + 1, 0] = af.max(n[:, 0])
    n_data[time_index + 1, 1] = af.max(n[:, 1])

    E1_ls = af.real(0.5 * (ls.N_q1 * ls.N_q2) 
                        * ifft2(ls.fields_solver.E1_hat)
                   )
   
    E_data[time_index + 1] = af.max(E1_ls)

h5f = h5py.File('data_ls.h5', 'w')
h5f.create_dataset('E', data = E_data)
h5f.create_dataset('n', data = n_data)
h5f.create_dataset('time', data = time_array)
h5f.close()
