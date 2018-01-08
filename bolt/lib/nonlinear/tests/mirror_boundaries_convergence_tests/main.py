import arrayfire as af
import numpy as np
import h5py
import pylab as pl

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 12, 7.5
pl.rcParams['figure.dpi']      = 300
pl.rcParams['image.cmap']      = 'jet'
pl.rcParams['lines.linewidth'] = 1.5
pl.rcParams['font.family']     = 'serif'
pl.rcParams['font.weight']     = 'bold'
pl.rcParams['font.size']       = 20
pl.rcParams['font.sans-serif'] = 'serif'
pl.rcParams['text.usetex']     = True
pl.rcParams['axes.linewidth']  = 1.5
pl.rcParams['axes.titlesize']  = 'medium'
pl.rcParams['axes.labelsize']  = 'medium'

pl.rcParams['xtick.major.size'] = 8
pl.rcParams['xtick.minor.size'] = 4
pl.rcParams['xtick.major.pad']  = 8
pl.rcParams['xtick.minor.pad']  = 8
pl.rcParams['xtick.color']      = 'k'
pl.rcParams['xtick.labelsize']  = 'medium'
pl.rcParams['xtick.direction']  = 'in'

pl.rcParams['ytick.major.size'] = 8
pl.rcParams['ytick.minor.size'] = 4
pl.rcParams['ytick.major.pad']  = 8
pl.rcParams['ytick.minor.pad']  = 8
pl.rcParams['ytick.color']      = 'k'
pl.rcParams['ytick.labelsize']  = 'medium'
pl.rcParams['ytick.direction']  = 'in'

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

# Time parameters:
t_final = 2.0

def time_evolution(nls):
    dt         = 0.01 * 32/nls.N_q1
    time_array = np.arange(dt, t_final + dt, dt)

    for time_index, t0 in enumerate(time_array):
        nls.strang_timestep(dt)

    return

N     = 2**np.arange(5, 10)
error = np.zeros(N.size)

for i in range(N.size):
    domain.N_q1 = int(N[i])
    # Defining the physical system to be solved:
    system = physical_system(domain,
                             boundary_conditions,
                             params,
                             initialize,
                             advection_terms,
                             collision_operator.BGK,
                             moment_defs
                            )

    # Declaring a linear system object which will 
    # evolve the defined physical system:
    nls = nonlinear_solver(system)
    n_nls_initial = nls.compute_moments('density')
    time_evolution(nls)
    n_nls    = nls.compute_moments('density')

    N_g      = nls.N_ghost
    error[i] = af.mean(af.abs(  n_nls[N_g:-N_g, N_g:-N_g] 
                              - n_nls_initial[N_g:-N_g, N_g:-N_g]
                             )
                      )
    
    pl.plot(n_nls[N_g:-N_g, 0])
    pl.plot(n_nls_initial[N_g:-N_g, 0], '--', color = 'black')
    pl.savefig(str(N[i])+'.png')
    pl.clf()

pl.loglog(N, error, 'o-', label = 'Numerical')
pl.loglog(N, error[0]*32/N, '--', color = 'black', 
          label = r'$O(N^{-1})$'
         )
pl.loglog(N, error[0]*32**2/N**2, '-.', color = 'black', 
          label = r'$O(N^{-2})$'
         )
pl.legend(loc = 'best')
pl.ylabel('Error')
pl.xlabel('$N$')
pl.savefig('convergence_plot.png')
