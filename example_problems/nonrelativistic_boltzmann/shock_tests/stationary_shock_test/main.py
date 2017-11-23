import arrayfire as af
import numpy as np
import h5py
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

# Declaring a linear system object which will evolve the defined physical system:
nls = nonlinear_solver(system)
nls.dump_moments('dump/0000')

dt      =   params.N_cfl * min(nls.dq1, nls.dq2) \
          / max(domain.p1_end,
                domain.p2_end,
                domain.p3_end
               )

time_array  = np.arange(dt, params.t_final + dt, dt)

for time_index, t0 in enumerate(time_array):
    nls.strang_timestep(dt)
    
    if(t0%0.01==0):
        nls.dump_moments('dump/%04d'%(time_index+1))
