import arrayfire as af
import numpy as np
import math
from petsc4py import PETSc
from mpi4py import MPI

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

PETSc.Sys.Print('With:', PETSc.COMM_WORLD.rank)

# Declaring a linear system object which will evolve the defined physical system:
nls = nonlinear_solver(system)
N_g = nls.N_ghost

# Time parameters:
dt = params.N_cfl * min(nls.dq1, nls.dq2) \
                  / max(domain.p1_end + domain.p2_end + domain.p3_end) # Combining the lists

nls.strang_timestep(dt)
af.sync()

tic = MPI.Wtime()

for i in range(100):
    nls.strang_timestep(dt)

af.eval(nls.f)
af.sync()

toc = MPI.Wtime()

PETSc.Sys.Print('Time:', toc - tic)
