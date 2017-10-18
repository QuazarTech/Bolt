import arrayfire as af
import numpy as np
import pylab as pl
import h5py

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


# Time parameters:
dt      = 0.001
t_final = 2.0

time_array = np.arange(dt, t_final + dt, dt)

def time_evolution(nls):

    for time_index, t0 in enumerate(time_array):
        
        if(time_index%100 == 0):
            print('Computing For Time =', t0)

        nls.strang_timestep(dt)

        n_nls = nls.compute_moments('density')

        p1_bulk_nls = nls.compute_moments('mom_p1_bulk') / n_nls
        p2_bulk_nls = nls.compute_moments('mom_p2_bulk') / n_nls
        p3_bulk_nls = nls.compute_moments('mom_p3_bulk') / n_nls

        T_nls = (  nls.compute_moments('energy')
                 - n_nls * p1_bulk_nls**2
                 - n_nls * p2_bulk_nls**2
                 - n_nls * p3_bulk_nls**2
                ) / n_nls
        
    h5f = h5py.File('numerical_' + str(domain.N_q1) + '.h5', 'w')
    h5f.create_dataset('temperature', data = T_nls[3:-3, 0])
    h5f.close()

N_x = np.array([32, 64, 128, 256, 512])  # No of spatial grid points

for i in range(N_x.size):
    
    print('For Nx =', N_x[i])

    domain.N_q1 = int(N_x[i])

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
    time_evolution(nls)
