import arrayfire as af
import numpy as np
import pylab as pl
import h5py
import time

from bolt.lib.physical_system import physical_system

from bolt.lib.nonlinear_solver.nonlinear_solver \
    import nonlinear_solver

from bolt.lib.linear_solver.linear_solver import linear_solver

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
dt      = 0.001
t_final = 2.0

time_array    = np.arange(0, t_final + dt, dt)
temp_data_nls = np.zeros_like(time_array)

def time_evolution():

    for time_index, t0 in enumerate(time_array):
        print('Computing For Time =', t0)

        n_nls = nls.compute_moments('density')

        p1_bulk_nls = nls.compute_moments('mom_p1_bulk') / n_nls
        p2_bulk_nls = nls.compute_moments('mom_p2_bulk') / n_nls
        p3_bulk_nls = nls.compute_moments('mom_p3_bulk') / n_nls

        E_nls = nls.compute_moments('energy')

        T_nls = (  nls.compute_moments('energy')
                 - n_nls * p1_bulk_nls**2
                 - n_nls * p2_bulk_nls**2
                 - n_nls * p3_bulk_nls**2
                ) / n_nls

        af.display(T_nls[:3, 3:-3])
        af.display(T_nls[-3:, 3:-3])

        time.sleep(5)
        temp_data_nls[time_index] = af.mean(T_nls[nls.N_ghost:-nls.N_ghost])
        nls.strang_timestep(dt)
        

time_evolution()

h5f = h5py.File('numerical.h5', 'w')
h5f.create_dataset('temperature', data = temp_data_nls)
h5f.create_dataset('time', data = time_array)
h5f.close()
