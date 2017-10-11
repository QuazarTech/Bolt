import arrayfire as af
import numpy as np
import pylab as pl

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

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 12, 7.5
pl.rcParams['figure.dpi']      = 100
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

# Time parameters:
dt      = 0.01
t_final = 2.0

time_array = np.arange(dt, t_final + dt, dt)

n_nls_initial = nls.compute_moments('density')

pl.contourf(np.array(nls.q1_center[3:-3, 3:-3]),
            np.array(nls.q2_center[3:-3, 3:-3]),
            np.array(n_nls_initial[3:-3, 3:-3]),
            100,
            cmap = 'gist_heat'
           )
pl.title('Time = 0')
pl.xlabel(r'$x$')
pl.ylabel(r'$y$')
pl.axes().set_aspect('equal')
pl.savefig('images/0000.png')
pl.clf()

def time_evolution():

    for time_index, t0 in enumerate(time_array):

        print('Computing For Time =', t0)
        nls.strang_timestep(dt)
        n_nls = nls.compute_moments('density')

        if((time_index+1)%1 == 0):

            pl.contourf(np.array(nls.q1_center[3:-3, 3:-3]),
                        np.array(nls.q2_center[3:-3, 3:-3]),
                        np.array(n_nls[3:-3, 3:-3]),
                        100,
                        cmap = 'gist_heat'
                       )
            pl.title('Time =' + str(t0))
            pl.xlabel(r'$x$')
            pl.ylabel(r'$y$')
            pl.axes().set_aspect('equal')
            pl.savefig('images/%04d'%((time_index+1)/1) + '.png')
            pl.clf()

time_evolution()
