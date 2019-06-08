import arrayfire as af
import numpy as np
import math
from petsc4py import PETSc
from mpi4py import MPI
MPI.WTIME_IS_GLOBAL=True

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
dt_fvm = params.N_cfl * min(nls.dq1, nls.dq2) \
                      / max(domain.p1_end + domain.p2_end + domain.p3_end) # joining elements of the list

dt_fdtd = params.N_cfl * min(nls.dq1, nls.dq2) \
                       / params.c # lightspeed

dt = min(dt_fvm, dt_fdtd)

if(params.t_restart == 0):
    time_elapsed = 0
    nls.dump_distribution_function('dump_f/t=0.000')
    nls.dump_moments('dump_moments/t=0.000')
    nls.dump_EM_fields('dump_fields/t=0.000')

else:
    time_elapsed = params.t_restart
    nls.load_distribution_function('dump_f/t=' + '%.3f'%time_elapsed)
    nls.load_EM_fields('dump_fields/t=' + '%.3f'%time_elapsed)

# Checking that the file writing intervals are greater than dt:
assert(params.dt_dump_f > dt)
assert(params.dt_dump_moments > dt)
assert(params.dt_dump_fields > dt)

print('Printing the minimum of the distribution functions for electrons and ions:')
print('f_min_electron:', af.min(nls.f[:, 0, :, :]))
print('f_min_ion:', af.min(nls.f[:, 1, :, :]))

print('MEAN DENSITY')
n = nls.compute_moments('density')
print('For electrons:', np.mean(np.array(n[:, 0, :, :])))
print('For ions:', np.mean(np.array(n[:, 1, :, :])))

print('MEAN ENERGY(PER UNIT MASS)')
E = nls.compute_moments('energy')
print('For electrons:', np.mean(np.array(E[:, 0, :, :])))
print('For ions:', np.mean(np.array(E[:, 1, :, :])))

#nls.strang_timestep(dt)
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
            nls.dump_moments('dump_moments/t=' + '%.3f'%time_elapsed)
            nls.dump_EM_fields('dump_fields/t=' + '%.3f'%time_elapsed)

    if(math.modf(time_elapsed/params.dt_dump_f)[0] < 1e-12):
        nls.dump_distribution_function('dump_f/t=' + '%.3f'%time_elapsed)

    PETSc.Sys.Print('Computing for time =', time_elapsed / params.t0, 't0, time taken = ', toc - tic, 'secs')

    timing_data.append(toc - tic)

# Writing the timing data to file:
#import h5py
#h5f = h5py.File('timing_data.h5', 'w')
#h5f.create_dataset('time_taken', data = np.array(timing_data))
#h5f.close()
