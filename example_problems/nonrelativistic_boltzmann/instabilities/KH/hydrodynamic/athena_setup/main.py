import arrayfire as af
import numpy as np
from petsc4py import PETSc

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

# Declaring a linear system object which will evolve
# the defined physical system:
nls = nonlinear_solver(system)
nls.dump_moments('dump/0000')

# Time parameters:
dt      =   params.cfl_number * min(nls.dq1, nls.dq2) \
          / max(params.p1_start,
                params.p2_start,
                params.p3_start
               )
          
t_final = params.t_final

time_array = np.arange(dt, t_final + dt, dt)

for time_index, t0 in enumerate(time_array):
    
    if((time_index+1)%100 == 0):
        PETSc.Sys.Print('Computing for Time =', t0)

    nls.strang_timestep(dt)

    if(t0%params.t_dump_moments == 0):
        nls.dump_moments('dump_moments/%04d'%time_index)

    if(t0%params.t_dump_f == 0):
        nls.dump_moments('dump_f/%04d'%time_index)
