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

# Declaring a nonlinear system object which will evolve the defined physical system:
nls = nonlinear_solver(system)
n_nls = nls.compute_moments('density')

params.rank = nls._comm.rank

# Time parameters:
dt      = 0.002
t_final = 1000.

time_array = np.arange(dt, t_final + dt, dt)
compute_electrostatic_fields(nls)
#params.mu = params.charge_electron*params.phi

N_g        = domain.N_ghost
#N_q1_local = params.mu.shape[0] - 2*N_g
#for i in range(N_g):
#    params.mu[i, :]                = 0.*params.mu[N_g, :]
#    params.mu[N_q1_local+N_g+i, :] = 0.*params.mu[N_q1_local+N_g-1, :]

print("rank = ", nls._comm.rank, " params.rank = ", params.rank,
      "mu = ", af.mean(params.mu[0, N_g:-N_g, N_g:-N_g]),
      "phi = ", af.mean(params.phi[0, N_g:-N_g, N_g:-N_g]),
      "density = ", af.mean(n_nls[0, N_g:-N_g, N_g:-N_g])
     )



#for time_step, t0 in enumerate(time_array):
#    PETSc.Sys.Print("Time step =", time_step, ", Time =", t0)
#
#    N_g = domain.N_ghost
#    mean_density = af.mean(n_nls[0, N_g:-N_g, N_g:-N_g])
#    density_pert = n_nls - mean_density
#    
#    dump_steps = 100
#    if (time_step%dump_steps==0):
#        nls.dump_moments('dumps/density_' + '%06d'%(time_step/dump_steps) + '.h5')
#        nls.dump_distribution_function('dumps/f_' + '%06d'%(time_step/dump_steps) + '.h5')
#
##    if (time_index%1==0):
##        pl.contourf(np.array(nls.q1_center)[0, N_g:-N_g, N_g:-N_g], \
##                    np.array(nls.q2_center)[0, N_g:-N_g, N_g:-N_g], \
##                    np.array(n_nls)[0, N_g:-N_g, N_g:-N_g], \
##                    100, cmap='bwr'
##                   )
##        pl.title('Time = ' + "%.2f"%(t0) )
##        pl.xlabel('$x$')
##        pl.ylabel('$y$')
##        pl.colorbar()
##        pl.gca().set_aspect('equal')
##        pl.savefig('/tmp/density_' + '%06d'%time_index + '.png' )
##        pl.clf()
##
##        pl.contourf(np.array(nls.q1_center)[0, N_g:-N_g, N_g:-N_g], \
##                    np.array(nls.q2_center)[0, N_g:-N_g, N_g:-N_g], \
##                    np.array(params.mu)[0, N_g:-N_g, N_g:-N_g], \
##                    100, cmap='bwr'
##                   )
##        pl.title('Time = ' + "%.2f"%(t0) )
##        pl.xlabel('$x$')
##        pl.ylabel('$y$')
##        pl.colorbar()
##        pl.gca().set_aspect('equal')
##        pl.savefig('/tmp/mu_' + '%06d'%time_index + '.png' )
##        pl.clf()
##    
###        pl.contourf(np.array(nls.q1_center), \
###                    np.array(nls.q2_center), \
###                    np.array(nls.E1), \
###                    100, cmap='bwr'
###                   )
###        pl.title('Time = ' + "%.2f"%(t0) )
###        pl.xlabel('$x$')
###        pl.ylabel('$y$')
###        pl.colorbar()
###        pl.gca().set_aspect('equal')
###        pl.savefig('/tmp/E1_' + '%06d'%time_index + '.png' )
###        pl.clf()
###
###        pl.contourf(np.array(nls.q1_center), \
###                    np.array(nls.q2_center), \
###                    np.array(nls.E2), \
###                    100, cmap='bwr'
###                   )
###        pl.title('Time = ' + "%.2f"%(t0) )
###        pl.xlabel('$x$')
###        pl.ylabel('$y$')
###        pl.colorbar()
###        pl.gca().set_aspect('equal')
###        pl.savefig('/tmp/E2_' + '%06d'%time_index + '.png' )
###        pl.clf()
###
##        f_at_desired_q = af.moddims(nls.f[:, N_g, N_g + nls.N_q2/2],
##                                    nls.N_p1, nls.N_p2
##                                   )
##        p1 = af.moddims(nls.p1, nls.N_p1, nls.N_p2)
##        p2 = af.moddims(nls.p2, nls.N_p1, nls.N_p2)
##        pl.contourf(np.array(p1), \
##                    np.array(p2), \
##                    np.array((f_at_desired_q)), \
##                    100, cmap='bwr'
##                   )
##        pl.title('Time = ' + "%.2f"%(t0) )
##        pl.xlabel('$x$')
##        pl.ylabel('$y$')
##        pl.colorbar()
##        pl.gca().set_aspect('equal')
##        pl.savefig('/tmp/f_' + '%06d'%time_index + '.png' )
##        pl.clf()
##
##    pl.contourf(np.array(params.mu)[N_g:-N_g, N_g:-N_g].transpose(), 100, cmap='jet')
##    pl.title('Time = ' + "%.2f"%(t0) )
##    pl.xlabel('$x$')
##    pl.ylabel('$y$')
##    pl.colorbar()
##    pl.gca().set_aspect('equal')
##    pl.savefig('/tmp/mu_' + '%06d'%time_index + '.png' )
##    pl.clf()
##
#
#    nls.strang_timestep(dt)
#
#    # Floors
#    nls.f     = af.select(nls.f < 1e-13, 1e-13, nls.f)
#    params.mu = af.select(params.mu < 1e-13, 1e-13, params.mu)
#
##    f_left = nls.boundary_conditions.\
##              f_left(nls.f, nls.q1_center, nls.q2_center,
##                      nls.p1, nls.p2, nls.p3, 
##                      nls.physical_system.params
##                     )
##    nls.f[:, -N_g:, 10:24] = f_right[:, -N_g:, 10:24]
#    n_nls = nls.compute_moments('density')
#    print("rank = ", nls._comm.rank, " params.rank = ", params.rank,
#          "mu = ", af.mean(params.mu[0, N_g:-N_g, N_g:-N_g]),
#          "phi = ", af.mean(params.phi[0, N_g:-N_g, N_g:-N_g]),
#          "density = ", mean_density
#         )
#    PETSc.Sys.Print("--------------------\n")

###pl.contourf(np.array(params.mu)[N_g:-N_g, N_g:-N_g].transpose(), 100, cmap='jet')
##pl.contourf(np.array(n_nls)[N_g:-N_g, N_g:-N_g].transpose(), 100, cmap='jet')
##pl.title('Density')
###pl.title('$\\mu$')
##pl.xlabel('$x$')
##pl.ylabel('$y$')
##pl.colorbar()
##pl.gca().set_aspect('equal')
##pl.show()
#
#phi_array = nls.poisson.glob_phi.getArray()
#phi_array = phi_array.reshape([nls.poisson.N_q3_3D_local, \
#                               nls.poisson.N_q2_3D_local, \
#                               nls.poisson.N_q1_3D_local]
#                             )
#pl.rcParams['figure.figsize']  = 20, 7.5
#pl.subplot(121)
#N_g = domain.N_ghost
#pl.contourf(
#            phi_array[nls.poisson.q3_2D_in_3D_index_start, :, :], 100, cmap='jet'
#           )
##pl.contourf(np.array(nls.E1)[N_g:-N_g, N_g:-N_g].transpose(), 100, cmap='jet'
##           )
#pl.colorbar()
#pl.title('Top View')
#pl.xlabel('$x$')
#pl.ylabel('$y$')
#pl.gca().set_aspect('equal')
#
#pl.subplot(122)
#pl.contourf(phi_array[:, nls.N_q2_poisson/2, :], 100, cmap='jet')
##pl.contourf(np.array(nls.E2)[N_g:-N_g, N_g:-N_g].transpose(), 100, cmap='jet')
#pl.title('Side View')
#pl.xlabel('$x$')
#pl.ylabel('$z$')
#pl.colorbar()
#pl.gca().set_aspect('equal')
#pl.show()
#
##h5f = h5py.File('dump/0000.h5', 'w')
##h5f.create_dataset('q1', data = nls.q1_center)
##h5f.create_dataset('q2', data = nls.q2_center)
##h5f.create_dataset('n', data = n_nls)
##h5f.close()
##
##def time_evolution():
##
##    for time_index, t0 in enumerate(time_array):
##        print('For Time =', t0    )
##        print('MIN(f) =', af.min(nls.f[3:-3, 3:-3]))
##        print('MAX(f) =', af.max(nls.f[3:-3, 3:-3]))
##        print('SUM(f) =', af.sum(nls.f[3:-3, 3:-3]))
##        print()
##
##        nls.strang_timestep(dt)
##        n_nls = nls.compute_moments('density')
##        
##        h5f = h5py.File('dump/%04d'%(time_index+1) + '.h5', 'w')
##        h5f.create_dataset('n', data = n_nls)
##        h5f.close()
##
##time_evolution()
