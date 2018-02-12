import arrayfire as af
import numpy as np
import h5py

from bolt.lib.physical_system import physical_system
from bolt.lib.nonlinear.nonlinear_solver import nonlinear_solver
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

N_g = system.N_ghost

# Declaring a linear system object which will evolve the defined physical system:
nls = nonlinear_solver(system)
ls  = linear_solver(system)

# Timestep as set by the CFL condition:
dt = params.N_cfl * min(nls.dq1, nls.dq2) \
                  / max(domain.p1_end, domain.p2_end, domain.p3_end)

time_array  = np.arange(0, params.t_final + dt, dt)

rho_data_nls = np.zeros([time_array.size, 2])
rho_data_ls  = np.zeros([time_array.size, 2])

E_data_nls = np.zeros([time_array.size])
E_data_ls  = np.zeros([time_array.size])

# Storing data at time t = 0:
n_nls              = nls.compute_moments('density')
rho_data_nls[0, 0] = af.max(n_nls[:, 0, N_g:-N_g, N_g:-N_g])
rho_data_nls[0, 1] = af.max(n_nls[:, 1, N_g:-N_g, N_g:-N_g])

n_ls              = ls.compute_moments('density')
rho_data_ls[0, 0] = af.max(n_ls[:, 0]) 
rho_data_ls[0, 1] = af.max(n_ls[:, 1]) 

E_data_nls[0] = af.max(nls.fields_solver.cell_centered_EM_fields[:, :, N_g:-N_g, N_g:-N_g])
E1_ls         = af.real(0.5 * (ls.N_q1 * ls.N_q2) 
                            * ifft2(ls.fields_solver.E1_hat)
                       )
E_data_ls[0]  = af.max(E1_ls)

nls.dump_distribution_function('dump_nls/0000')
ls.dump_distribution_function('dump_ls/0000')

for time_index, t0 in enumerate(time_array[1:]):

    print('Computing For Time =', t0)
    
    nls.strang_timestep(dt)
    ls.RK4_timestep(dt)
    
    nls.dump_distribution_function('dump_nls/%04d'%(time_index+1))
    ls.dump_distribution_function('dump_ls/%04d'%(time_index+1))

    n_nls                           = nls.compute_moments('density')
    rho_data_nls[time_index + 1, 0] = af.max(n_nls[:, 0, N_g:-N_g, N_g:-N_g])
    rho_data_nls[time_index + 1, 1] = af.max(n_nls[:, 1, N_g:-N_g, N_g:-N_g])
    
    n_ls                           = ls.compute_moments('density')
    rho_data_ls[time_index + 1, 0] = af.max(n_ls[:, 0]) 
    rho_data_ls[time_index + 1, 1] = af.max(n_ls[:, 1]) 

    E_data_nls[time_index] = af.max(nls.fields_solver.cell_centered_EM_fields[:, :, N_g:-N_g, N_g:-N_g])
    E1_ls                  = af.real(0.5 * (ls.N_q1 * ls.N_q2) 
                                         * ifft2(ls.fields_solver.E1_hat)
                                    )
    E_data_ls[time_index]  = af.max(E1_ls)

h5f = h5py.File('data.h5', 'w')
h5f.create_dataset('E_ls', data = E_data_ls)
h5f.create_dataset('E_nls', data = E_data_nls)
h5f.create_dataset('n_ls', data = rho_data_ls)
h5f.create_dataset('n_nls', data = rho_data_nls)
h5f.create_dataset('time', data = time_array)
h5f.close()
