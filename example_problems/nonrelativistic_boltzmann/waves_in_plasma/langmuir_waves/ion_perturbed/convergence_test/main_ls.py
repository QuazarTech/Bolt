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

for i in range(N.size):
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

    for time_index, t0 in enumerate(time_array[1:]):
        print('Computing For Time =', t0)
        ls.RK5_timestep(dt)

    h5f = h5py.File('dump/ls_N_%04d'%(int(N[i])) + '.h5', 'w')
    h5f.create_dataset('f_hat', data = np.array(ls.f_hat))
    h5f.close()
