import arrayfire as af
import numpy as np
import h5py
import matplotlib as mpl
mpl.use('agg')
import pylab as pl

from bolt.lib.physical_system import physical_system
from bolt.lib.nonlinear.nonlinear_solver import nonlinear_solver

import domain
import boundary_conditions
import params
import initialize

import bolt.src.nonrelativistic_boltzmann.advection_terms as advection_terms
import bolt.src.nonrelativistic_boltzmann.collision_operator as collision_operator
import bolt.src.nonrelativistic_boltzmann.moments as moments

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
                         moments
                        )

N_g = system.N_ghost


# Declaring a linear system object which will evolve the defined physical system:
nls = nonlinear_solver(system)

# Timestep as set by the CFL condition:
dt = params.N_cfl * min(nls.dq1, nls.dq2) \
                  / max(domain.p1_end, domain.p2_end, domain.p3_end)

time_array = np.arange(0, params.t_final + dt, dt)
data       = np.zeros(time_array.size)

data[0] = 0

for time_index, t0 in enumerate(time_array[1:]):

    rho_n       = -1 * nls.compute_moments('density')
    rho_n[0, 1] = -1 * rho_n[0, 1]
    rho_n       = af.sum(rho_n, 1)

    nls.strang_timestep(dt)

    rho_n_plus_one       = -1 * nls.compute_moments('density')
    rho_n_plus_one[0, 1] = -1 * rho_n_plus_one[0, 1]
    rho_n_plus_one       = af.sum(rho_n_plus_one, 1)

    # divE      = nls.fields_solver.compute_divE()
    drho_dt = (rho_n_plus_one - rho_n) / dt

    data[time_index + 1] = af.mean(af.abs(drho_dt + nls.fields_solver.J1)[:, :, N_g:-N_g, N_g:-N_g])

    if(time_index % 10 == 0):
        pl.plot(np.array(nls.q1_center[:, :, 3:-3, 0]).ravel(),
                np.array(nls.compute_moments('density')[:, 0, 3:-3, 0]).ravel(),
                label = 'Electrons'
               )
        pl.plot(np.array(nls.q1_center[:, :, 3:-3, 0]).ravel(),
                np.array(nls.compute_moments('density')[:, 1, 3:-3, 0]).ravel(),
                '--', color = 'C3',
                label = 'Positrons'
               )
        pl.ylim([0, 0.01])
        pl.ylabel(r'$n$')
        pl.xlabel(r'$x$')
        pl.legend()
        pl.title('Time = %.2f'%t0)
        pl.savefig('images/%04d'%(time_index / 10) + '.png')
        pl.clf()

pl.plot(time_array, data)
pl.ylabel('Error')
pl.xlabel('Time')
pl.savefig('plot.png')
pl.clf()

pl.semilogy(time_array, data)
pl.ylabel('Error')
pl.xlabel('Time')
pl.savefig('semilogyplot.png')
pl.clf()
