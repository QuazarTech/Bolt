import arrayfire as af
import numpy as np
from petsc4py import PETSc

from bolt.lib.physical_system import physical_system
from bolt.lib.nonlinear.nonlinear_solver import nonlinear_solver

import domain
import boundary_conditions
import initialize
import params

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
dt = params.N_cfl * params.t0 * min(nls.dq1, nls.dq2) \
                              / max(domain.p1_end, domain.p2_end, domain.p3_end)

time_array = np.arange(0, params.t_final + dt, dt)

for time_index, t0 in enumerate(time_array):

    nls.dump_moments('dump_moments/%04d'%time_index)

    if(time_index % 10 == 0):
        PETSc.Sys.Print('Computing For Time =', t0)
    
    nls.strang_timestep(dt)
