import sys
import arrayfire as af
import numpy as np
import pylab as pl
import h5py
import petsc4py, sys; petsc4py.init(sys.argv)
from petsc4py import PETSc

from bolt.lib.physical_system import physical_system

from bolt.lib.nonlinear_solver.nonlinear_solver \
    import nonlinear_solver
from bolt.lib.nonlinear_solver.EM_fields_solver.electrostatic \
    import compute_electrostatic_fields

import domain
import boundary_conditions
import params
import initialize

import bolt.src.electronic_boltzmann.advection_terms as advection_terms

import bolt.src.electronic_boltzmann.collision_operator \
    as collision_operator

import bolt.src.electronic_boltzmann.moment_defs as moment_defs

#pl.rcParams['figure.figsize']  = 12, 7.5
#pl.rcParams['figure.dpi']      = 150
#pl.rcParams['image.cmap']      = 'jet'
#pl.rcParams['lines.linewidth'] = 1.5
#pl.rcParams['font.family']     = 'serif'
#pl.rcParams['font.weight']     = 'bold'
#pl.rcParams['font.size']       = 20
#pl.rcParams['font.sans-serif'] = 'serif'
#pl.rcParams['text.usetex']     = False
#pl.rcParams['axes.linewidth']  = 1.5
#pl.rcParams['axes.titlesize']  = 'medium'
#pl.rcParams['axes.labelsize']  = 'medium'
#
#pl.rcParams['xtick.major.size'] = 8
#pl.rcParams['xtick.minor.size'] = 4
#pl.rcParams['xtick.major.pad']  = 8
#pl.rcParams['xtick.minor.pad']  = 8
#pl.rcParams['xtick.color']      = 'k'
#pl.rcParams['xtick.labelsize']  = 'medium'
#pl.rcParams['xtick.direction']  = 'in'
#
#pl.rcParams['ytick.major.size'] = 8
#pl.rcParams['ytick.minor.size'] = 4
#pl.rcParams['ytick.major.pad']  = 8
#pl.rcParams['ytick.minor.pad']  = 8
#pl.rcParams['ytick.color']      = 'k'
#pl.rcParams['ytick.labelsize']  = 'medium'
#pl.rcParams['ytick.direction']  = 'in'


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

    viewer = PETSc.Viewer().createHDF5(params.phi_restart_file, 
                                       PETSc.Viewer.Mode.READ, 
                                       comm=nls._comm
                                      )
    PETSc.Object.setName(nls.poisson.glob_phi, 'phi')
    nls.poisson.glob_phi.load(viewer)

    params.solve_for_equilibrium = 0
    compute_electrostatic_fields(nls)

    params.mu = params.global_chem_potential + params.charge_electron*params.phi
    params.mu_analytic = params.global_chem_potential + params.charge_electron*params.phi
    params.mu = af.moddims(params.mu,
                           1, 
                           nls.N_q1_local + 2*N_g, 
                           nls.N_q2_local + 2*N_g
                          )
else:
    if (params.solve_for_equilibrium):
        PETSc.Sys.Print("=============================")
        PETSc.Sys.Print("Solving for equilibrium state")
        PETSc.Sys.Print("=============================")
        params.solve_for_equilibrium = 1
        compute_electrostatic_fields(nls)
        params.solve_for_equilibrium = 0
        
        params.mu = params.global_chem_potential + params.charge_electron*params.phi
        params.mu = af.moddims(params.mu,
                               1, 
                               nls.N_q1_local + 2*N_g, 
                               nls.N_q2_local + 2*N_g
                              )
    
        nls.f   = params.fermi_dirac(params.mu, params.E_band)
    
        PETSc.Sys.Print("=====================")
        PETSc.Sys.Print("Solution diagnostics:")
        PETSc.Sys.Print("=====================")
        
        density = nls.compute_moments('density')
        print("rank = ", params.rank, "\n",
              "     <mu>    = ", af.mean(params.mu[0, N_g:-N_g, N_g:-N_g]), "\n",
              "     max(mu) = ", af.max(params.mu[0, N_g:-N_g, N_g:-N_g]), "\n",
              #"     <phi>   = ", af.mean(params.phi[N_g:-N_g, N_g:-N_g]), "\n",
              "     <n>     = ", af.mean(density[0, N_g:-N_g, N_g:-N_g]), "\n",
              "     max(n)  = ", af.max(density[0, N_g:-N_g, N_g:-N_g]), "\n",
              "     |E1|    = ", af.mean(af.abs(nls.cell_centered_EM_fields[0, N_g:-N_g, N_g:-N_g])),
              "\n",
              "     |E2|    = ", af.mean(af.abs(nls.cell_centered_EM_fields[1, N_g:-N_g, N_g:-N_g]))
             )
    
        PETSc.Sys.Print("===============")
        PETSc.Sys.Print("Dumping data...")
        PETSc.Sys.Print("===============")
        
        nls.dump_moments('dumps/moments_eqbm')
        nls.dump_distribution_function('dumps/f_eqbm')
    
        # Dump EM fields
        af.flat(nls.cell_centered_EM_fields[:, N_g:-N_g, N_g:-N_g]).to_ndarray(nls._glob_fields_array)
        PETSc.Object.setName(nls._glob_fields, 'fields')
        viewer = PETSc.Viewer().createHDF5('dumps/fields_eqbm.h5',
                                           'w', comm=nls._comm
                                          )
        viewer(nls._glob_fields)
    
        # Dump eqbm potential
        PETSc.Object.setName(nls.poisson.glob_phi, 'phi')
        viewer = PETSc.Viewer().createHDF5('dumps/phi_eqbm.h5',
                                           'w', comm=nls._comm
                                          )
        viewer(nls.poisson.glob_phi)
    
        sys.exit("Terminating execution")
    else:

        #compute_electrostatic_fields(nls)
        pass

density = nls.compute_moments('density')
print("rank = ", params.rank, "\n",
      "     <mu>    = ", af.mean(params.mu[0, N_g:-N_g, N_g:-N_g]), "\n",
      "     max(mu) = ", af.max(params.mu[0, N_g:-N_g, N_g:-N_g]), "\n",
      #"     <phi>   = ", af.mean(params.phi[N_g:-N_g, N_g:-N_g]), "\n",
      "     <n>     = ", af.mean(density[0, N_g:-N_g, N_g:-N_g]), "\n",
      "     max(n)  = ", af.max(density[0, N_g:-N_g, N_g:-N_g]), "\n",
      "     |E1|    = ", af.mean(af.abs(nls.cell_centered_EM_fields[0, N_g:-N_g, N_g:-N_g])),
      "\n",
      "     |E2|    = ", af.mean(af.abs(nls.cell_centered_EM_fields[1, N_g:-N_g, N_g:-N_g]))
     )

nls.f     = af.select(nls.f < 1e-20, 1e-20, nls.f)
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
        file_number = '%06d'%dump_counter
        dump_counter= dump_counter + 1
        dump_time_array.append(params.current_time)
        PETSc.Sys.Print("=====================================================")
        PETSc.Sys.Print("Dumping data at time step =", time_step,
                         ", file number =", file_number
                       )
        PETSc.Sys.Print("=====================================================")
        nls.dump_moments('dumps/moments_' + file_number)
        #nls.dump_distribution_function('dumps/f_' + file_number)
        nls.dump_aux_arrays([params.mu,
                             params.mu_ee,
                             params.T_ee,
                             params.vel_drift_x, params.vel_drift_y],
                             'lagrange_multipliers',
                             'dumps/lagrange_multipliers_' + file_number
                            )

        # Dump EM fields
        af.flat(nls.cell_centered_EM_fields[:, N_g:-N_g, N_g:-N_g]).to_ndarray(nls._glob_fields_array)
        PETSc.Object.setName(nls._glob_fields, 'fields')
        viewer = PETSc.Viewer().createHDF5('dumps/fields_' 
                                           + '%06d'%(time_step/dump_steps) + '.h5',
                                           'w', comm=nls._comm
                                          )
        viewer(nls._glob_fields)


    dt_force_constraint = 0.
#    dt_force_constraint = \
#        0.5 * np.min(nls.dp1, nls.dp2) \
#            / np.max((af.max(nls.cell_centered_EM_fields[0]),
#                      af.max(nls.cell_centered_EM_fields[1])
#                     )
#                    )


    PETSc.Sys.Print("Time step =", time_step, ", Time =", t0)

    mean_density = af.mean(density[0, N_g:-N_g, N_g:-N_g])
    density_pert = density - mean_density
    
#    if (time_step%1==0):
#        chemical_potential =  np.array(params.mu)[0, :, :] \
#                            - params.charge_electron*np.array(params.phi)[:, :]
#        pl.contourf(np.array(nls.q1_center)[0, N_g:-N_g, N_g:-N_g], \
#                    np.array(nls.q2_center)[0, N_g:-N_g, N_g:-N_g], \
#                    np.array(params.mu)[0, N_g:-N_g, N_g:-N_g], \
#                    100, cmap='bwr'
#                   )
#        pl.title('Time = ' + "%.2f"%(t0) )
#        pl.xlabel('$x$')
#        pl.ylabel('$y$')
#        pl.colorbar()
#        pl.gca().set_aspect('equal')
#        pl.savefig('/tmp/mu_' + '%06d'%time_step + '.png' )
#        pl.clf()
#
#        pl.contourf(np.array(nls.q1_center)[0, N_g:-N_g, N_g:-N_g], \
#                    np.array(nls.q2_center)[0, N_g:-N_g, N_g:-N_g], \
#                    np.array(density)[0, N_g:-N_g, N_g:-N_g], \
#                    100, cmap='bwr'
#                   )
#        pl.title('Time = ' + "%.2f"%(t0) )
#        pl.xlabel('$x$')
#        pl.ylabel('$y$')
#        pl.colorbar()
#        pl.gca().set_aspect('equal')
#        pl.savefig('/tmp/density_' + '%06d'%time_step + '.png' )
#        pl.clf()
#
#        pl.contourf(np.array(nls.q1_center)[0, N_g:-N_g, N_g:-N_g], \
#                    np.array(nls.q2_center)[0, N_g:-N_g, N_g:-N_g], \
#                    np.array(params.phi)[N_g:-N_g, N_g:-N_g], \
#                    100, cmap='bwr'
#                   )
#        pl.title('Time = ' + "%.2f"%(t0) )
#        pl.xlabel('$x$')
#        pl.ylabel('$y$')
#        pl.colorbar()
#        pl.gca().set_aspect('equal')
#        pl.savefig('/tmp/phi_' + '%06d'%time_step + '.png' )
#        pl.clf()
#    
#        pl.contourf(np.array(nls.q1_center)[0, N_g:-N_g, N_g:-N_g], \
#                    np.array(nls.q2_center)[0, N_g:-N_g, N_g:-N_g], \
#                    np.array(nls.cell_centered_EM_fields)[0, N_g:-N_g, N_g:-N_g], \
#                    100, cmap='bwr'
#                   )
#        pl.title('Time = ' + "%.2f"%(t0) )
#        pl.xlabel('$x$')
#        pl.ylabel('$y$')
#        pl.colorbar()
#        pl.gca().set_aspect('equal')
#        pl.savefig('/tmp/E1_' + '%06d'%time_step + '.png' )
#        pl.clf()
#
#        pl.contourf(np.array(nls.q1_center)[0, N_g:-N_g, N_g:-N_g], \
#                    np.array(nls.q2_center)[0, N_g:-N_g, N_g:-N_g], \
#                    np.array(nls.cell_centered_EM_fields)[1, N_g:-N_g, N_g:-N_g], \
#                    100, cmap='bwr'
#                   )
#        pl.title('Time = ' + "%.2f"%(t0) )
#        pl.xlabel('$x$')
#        pl.ylabel('$y$')
#        pl.colorbar()
#        pl.gca().set_aspect('equal')
#        pl.savefig('/tmp/E2_' + '%06d'%time_step + '.png' )
#        pl.clf()
#
#        f_at_desired_q = af.moddims(nls.f[:, N_g, N_g + 0.*nls.N_q2/2],
#                                    nls.N_p1, nls.N_p2
#                                   )
#        p1 = af.moddims(nls.p1, nls.N_p1, nls.N_p2)
#        p2 = af.moddims(nls.p2, nls.N_p1, nls.N_p2)
#        pl.contourf(np.array(p1), \
#                    np.array(p2), \
#                    np.array((f_at_desired_q)), \
#                    100, cmap='bwr'
#                   )
#        pl.title('Time = ' + "%.2f"%(t0) )
#        pl.xlabel('$x$')
#        pl.ylabel('$y$')
#        pl.colorbar()
#        pl.gca().set_aspect('equal')
#        pl.savefig('/tmp/f_' + '%06d'%time_step + '.png' )
#        pl.clf()

    nls.strang_timestep(dt)
    t0                  = t0 + dt
    time_step           = time_step + 1
    params.time_step    = time_step
    params.current_time = t0

    # Floors
    nls.f     = af.select(nls.f < 1e-20, 1e-20, nls.f)

    density = nls.compute_moments('density')
    print("rank = ", params.rank, "\n",
          "     <mu>    = ", af.mean(params.mu[0, N_g:-N_g, N_g:-N_g]), "\n",
          "     max(mu) = ", af.max(params.mu[0, N_g:-N_g, N_g:-N_g]), "\n",
          #"     <phi>   = ", af.mean(params.phi[N_g:-N_g, N_g:-N_g]), "\n",
          "     <n>     = ", af.mean(density[0, N_g:-N_g, N_g:-N_g]), "\n",
          "     max(n)  = ", af.max(density[0, N_g:-N_g, N_g:-N_g]), "\n",
          "     |E1|    = ", af.mean(af.abs(nls.cell_centered_EM_fields[0, N_g:-N_g, N_g:-N_g])),
          "\n",
          "     |E2|    = ", af.mean(af.abs(nls.cell_centered_EM_fields[1, N_g:-N_g, N_g:-N_g]))
         )
    PETSc.Sys.Print("--------------------\n")

if (params.rank==0):
    np.savetxt("dump_time_array.txt", dump_time_array)

#phi_array = nls.poisson.glob_phi.getArray()
#phi_array = phi_array.reshape([nls.poisson.N_q3_3D_local, \
#                               nls.poisson.N_q2_3D_local, \
#                               nls.poisson.N_q1_3D_local]
#                             )
#
#q3_index = np.where(nls.poisson.q3_3D[N_g:-N_g] >= nls.location_in_q3)[0][0]
#pl.plot(nls.poisson.q1_3D[N_g:-N_g],
#        phi_array[q3_index, nls.poisson.N_q2_3D_local/2, :], 'o-'
#       )
#pl.show()

#pl.rcParams['figure.figsize']  = 20, 7.5
#pl.subplot(121)
#N_g = domain.N_ghost
#pl.contourf(
#            phi_array[nls.N_q3_poisson/2, :, :], 100, cmap='jet'
#           )
#pl.colorbar()
#pl.title('Top View')
#pl.xlabel('$x$')
#pl.ylabel('$y$')
#pl.gca().set_aspect('equal')
#
#pl.subplot(122)
#pl.contourf(phi_array[:, nls.N_q2_poisson/2, :], 100, cmap='jet')
#pl.title('Side View')
#pl.xlabel('$x$')
#pl.ylabel('$z$')
#pl.colorbar()
#pl.gca().set_aspect('equal')
#pl.show()
