import arrayfire as af
import numpy as np
import math
from petsc4py import PETSc
from mpi4py import MPI
MPI.WTIME_IS_GLOBAL=True

from bolt.lib.physical_system import physical_system
from bolt.lib.nonlinear.nonlinear_solver import nonlinear_solver
from bolt.lib.utils.parallel_reduction_ops import global_min, global_mean

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
dt_fvm = params.N_cfl * min(nls.dq1, nls.dq2) \
                      / max(domain.p1_end + domain.p2_end + domain.p3_end) # joining elements of the list

dt_fdtd = params.N_cfl * min(nls.dq1, nls.dq2) \
                       / params.c # lightspeed

dt = min(dt_fvm, dt_fdtd)

if(params.t_restart == 0):
    time_elapsed = 0
    nls.dump_distribution_function('dump_f/t=0.000000')
    nls.dump_moments('dump_moments/t=0.000000')
    nls.dump_EM_fields('dump_fields/t=0.000000')

else:
    time_elapsed = params.t_restart
    nls.load_distribution_function('dump_f/t=' + '%.6f'%time_elapsed)
    nls.load_EM_fields('dump_fields/t=' + '%.6f'%time_elapsed)

# Checking that the file writing intervals are greater than dt:
assert(params.dt_dump_f > dt)
assert(params.dt_dump_moments > dt)
assert(params.dt_dump_fields > dt)

PETSc.Sys.Print('\nMinimum of the distribution functions for electrons and ions:')
PETSc.Sys.Print('Electrons:', global_min(nls.f[:, 0, :, :]))
PETSc.Sys.Print('Ions     :', global_min(nls.f[:, 1, :, :]))
PETSc.Sys.Print('\n')

PETSc.Sys.Print('MEAN DENSITY')
n = nls.compute_moments('density')
PETSc.Sys.Print('Electrons:', global_mean(n[:, 0, :, :]))
PETSc.Sys.Print('Ions     :', global_mean(n[:, 1, :, :]))
PETSc.Sys.Print('\n')

PETSc.Sys.Print('MEAN ENERGY (PER UNIT MASS)')
E = nls.compute_moments('energy')
PETSc.Sys.Print('Electrons:', global_mean(E[:, 0, :, :]))
PETSc.Sys.Print('Ions     :', global_mean(E[:, 1, :, :]))
PETSc.Sys.Print('\n')

timing_data = []
while(abs(time_elapsed - params.t_final) > 1e-12):

    tic = MPI.Wtime()
    nls.strang_timestep(dt)
    toc = MPI.Wtime()

    time_elapsed += dt

    if(params.dt_dump_moments != 0):
        # We step by delta_dt to get the values at dt_dump
        delta_dt =   (1 - math.modf(time_elapsed/params.dt_dump_moments)[0]) \
                   * params.dt_dump_moments

        if((delta_dt-dt)<1e-5):
            nls.strang_timestep(delta_dt)
            time_elapsed += delta_dt
            nls.dump_moments('dump_moments/t=' + '%.6f'%time_elapsed)
            nls.dump_EM_fields('dump_fields/t=' + '%.6f'%time_elapsed)

    if(math.modf(time_elapsed/params.dt_dump_f)[0] < 1e-12):
        nls.dump_distribution_function('dump_f/t=' + '%.6f'%time_elapsed)

    PETSc.Sys.Print('Time =', format(time_elapsed / params.t0, '.4f'),
                    't0, dt = ', format(dt / params.t0, '.4f'),
                    't0, time taken = ', format(toc - tic, '.4f'), 'secs'
                   )

    timing_data.append(toc - tic)
