import arrayfire as af
import numpy as np
import h5py

from bolt.lib.physical_system import physical_system
from bolt.lib.linear.linear_solver import linear_solver
from bolt.lib.utils.fft_funcs import ifft2

import sys
sys.path.append('../')

import domain
import boundary_conditions
import params
import initialize

import bolt.src.nonrelativistic_boltzmann.advection_terms as advection_terms
import bolt.src.nonrelativistic_boltzmann.collision_operator as collision_operator
import bolt.src.nonrelativistic_boltzmann.moments as moments

N = 2**np.arange(5, 11)
for i in range(N.size):
    domain.N_q1 = int(N[i])
    domain.N_p1 = int(N[i])
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

    t_final    = 0.1
    time_array = np.arange(0, t_final + dt, dt)
    
    for time_index, t0 in enumerate(time_array[1:]):
        print('Computing For Time =', t0)
        ls.RK5_timestep(dt)

    ls.dump_distribution_function('dump/ls_N_%04d'%int(N[i]))
