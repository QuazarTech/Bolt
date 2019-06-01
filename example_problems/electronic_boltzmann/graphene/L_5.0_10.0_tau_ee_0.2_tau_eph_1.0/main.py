import sys
import arrayfire as af
import numpy as np
import pylab as pl
import h5py
import petsc4py, sys; petsc4py.init(sys.argv)
from petsc4py import PETSc

from bolt.lib.physical_system import physical_system

from bolt.lib.nonlinear.nonlinear_solver \
    import nonlinear_solver
#from bolt.lib.nonlinear.fields.fields.fields \
#    import fields_solver.compute_electrostatic_fields

import domain
import boundary_conditions
import params
import initialize

import bolt.src.electronic_boltzmann.advection_terms as advection_terms

import bolt.src.electronic_boltzmann.collision_operator \
    as collision_operator

import bolt.src.electronic_boltzmann.moment_defs as moment_defs


# Defining the physical system to be solved:
system = physical_system(domain,
                         boundary_conditions,
                         params,
                         initialize,
                         advection_terms,
                         collision_operator.RTA,
                         moment_defs
                        )

# Time parameters:
dt      = params.dt
t_final = params.t_final
params.current_time = t0        = 0.0
params.time_step    = time_step = 0
dump_counter = 0
dump_time_array = []

N_g        = domain.N_ghost

# Declaring a nonlinear system object which will evolve the defined physical system:
nls = nonlinear_solver(system)
params.rank = nls._comm.rank

if (params.restart):
    nls.load_distribution_function(params.restart_file)

density = nls.compute_moments('density')
print("rank = ", params.rank, "\n",
      "     <mu>    = ", af.mean(params.mu[0, 0, N_g:-N_g, N_g:-N_g]), "\n",
      "     max(mu) = ", af.max(params.mu[0, 0, N_g:-N_g, N_g:-N_g]), "\n",
      "     <n>     = ", af.mean(density[0, 0, N_g:-N_g, N_g:-N_g]), "\n",
      "     max(n)  = ", af.max(density[0, 0, N_g:-N_g, N_g:-N_g]), "\n"
     )

nls.f = af.select(nls.f < 1e-20, 1e-20, nls.f)
while t0 < t_final:

    # Refine to machine error
    if (time_step==0):
        params.collision_nonlinear_iters = 10
    else:
        params.collision_nonlinear_iters = params.collision_operator_nonlinear_iters

    dump_steps = params.dump_steps
    # Uncomment if need to dump more frequently during a desired time interval
    #if (params.current_time > 149. and params.current_time < 154):
    #    dump_steps = 1
    #else:
    #    dump_steps = params.dump_steps
    if (time_step%dump_steps==0):
        file_number  = '%06d'%dump_counter
        dump_counter = dump_counter + 1
        dump_time_array.append(params.current_time)
        PETSc.Sys.Print("=====================================================")
        PETSc.Sys.Print("Dumping data at time step =", time_step,
                         ", file number =", file_number
                       )
        PETSc.Sys.Print("=====================================================")
        if (params.rank==0):
            np.savetxt("dump_time_array.txt", dump_time_array)

        nls.dump_moments('dumps/moments_' + file_number)

        if (time_step==0):
            nls.dump_distribution_function('dumps/f_' + file_number)
        
        #nls.dump_distribution_function('dumps/f_' + file_number)
        nls.dump_aux_arrays([params.mu,
                             params.mu_ee,
                             params.T_ee,
                             params.vel_drift_x, params.vel_drift_y],
                             'lagrange_multipliers',
                             'dumps/lagrange_multipliers_' + file_number
                            )

    dt_force_constraint = 0.
#    dt_force_constraint = \
#        0.5 * np.min(nls.dp1, nls.dp2) \
#            / np.max((af.max(nls.cell_centered_EM_fields[0]),
#                      af.max(nls.cell_centered_EM_fields[1])
#                     )
#                    )

    PETSc.Sys.Print("Time step =", time_step, ", Time =", t0)

    nls.strang_timestep(dt)
    t0                  = t0 + dt
    time_step           = time_step + 1
    params.time_step    = time_step
    params.current_time = t0

    # Floors
    nls.f     = af.select(nls.f < 1e-20, 1e-20, nls.f)

    density = nls.compute_moments('density')
    print("rank = ", params.rank, "\n",
          "     <mu>    = ", af.mean(params.mu[0, 0, N_g:-N_g, N_g:-N_g]), "\n",
          "     max(mu) = ", af.max(params.mu[0, 0, N_g:-N_g, N_g:-N_g]), "\n",
          "     <n>     = ", af.mean(density[0, 0, N_g:-N_g, N_g:-N_g]), "\n",
          "     max(n)  = ", af.max(density[0, 0, N_g:-N_g, N_g:-N_g]), "\n"
         )
    PETSc.Sys.Print("--------------------\n")

nls.dump_distribution_function('dumps/f_laststep')
