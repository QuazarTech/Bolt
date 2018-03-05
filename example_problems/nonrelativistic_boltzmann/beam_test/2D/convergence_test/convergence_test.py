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

N     = 2**np.arange(5, 8)
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
                             moments
                            )

    # Declaring a linear system object which will evolve the defined physical system:
    nls = nonlinear_solver(system)

    # Time parameters:
    dt      = 0.001 * 32/int(N[i])
    t_final = 1.0

    time_array = np.arange(dt, t_final + dt, dt)

    N_lower = int(round(0.4 * int(round(np.sqrt(nls.q1_center.elements())))))
    N_upper = int(round(0.6 * int(round(np.sqrt(nls.q1_center.elements())))))

    mult        = lambda a,b:a*b
    f_reference = 0 * nls.f
    
    f_reference[3, :, :, N_lower:N_upper] = \
        af.broadcast(mult, nls.f**0, af.exp(-250 * (nls.q2_center - 0.5)**2))[3, :, :, N_lower:N_upper] 

    for time_index, t0 in enumerate(time_array):
        print('Computing for Time =', t0)
        nls.strang_timestep(dt)

    error[i] = af.mean(af.abs(nls.f[:, :, 3:-3, 3:-3] - f_reference[:, :, 3:-3, 3:-3]))

print(error)
print(np.polyfit(np.log10(N), np.log10(error), 1))
