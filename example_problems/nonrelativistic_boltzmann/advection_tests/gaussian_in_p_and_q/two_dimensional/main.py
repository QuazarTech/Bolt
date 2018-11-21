import arrayfire as af
import numpy as np
import h5py
import math
from petsc4py import PETSc

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
dt      = 0.001 * 32/nls.N_q1
t_final = params.t_final

time_array  = np.arange(dt, t_final + dt, dt)

print('Printing the minimum and maximum of the distribution functions :')
print('f_min:', af.min(nls.f))
print('f_max:', af.max(nls.f))

# Dumping the distribution function at t = 0:
# nls.dump_distribution_function('dump_f/t=0.000')
nls.dump_moments('dump_moments/t=0.000')

for time_index, t0 in enumerate(time_array):
    
    nls.strang_timestep(dt)
    PETSc.Sys.Print('Computing For Time =', t0)

    delta_dt =   (1 - math.modf(t0/params.dt_dump_moments)[0]) \
               * params.dt_dump_moments

    nls.dump_moments('dump_moments/t=' + '%.3f'%t0)
    # nls.dump_distribution_function('dump_f/t=' + '%.3f'%t0)

    # if((delta_dt-dt) < 1e-5):
    #     nls.dump_distribution_function('dump_f/t=' + '%.3f'%t0)
    #     nls.dump_moments('dump_moments/t=' + '%.3f'%t0)
