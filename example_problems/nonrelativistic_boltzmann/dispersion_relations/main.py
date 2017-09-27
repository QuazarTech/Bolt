import arrayfire as af
import numpy as np
import pylab as pl
import petsc4py
import sys
petsc4py.init(sys.argv)

from bolt.lib.physical_system import physical_system
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
ls  = linear_solver(system)

# Time parameters:
dt      = 0.002
t_final = 0.8

time_array = np.arange(0, t_final + dt, dt)

# Initializing Array used in storing the data:
rho_data = np.zeros_like(time_array)

for time_index, t0 in enumerate(time_array):
    # print('Computing For Time =', t0)

    n = ls.compute_moments('density')
    rho_data[time_index] = af.max(n)
    ls.RK2_timestep(dt)

pl.plot(time_array, rho_data)
pl.ylabel(r'MAX($\rho$)')
pl.xlabel('Time')
pl.savefig('rho.png')
pl.clf()

f_hat = abs(np.fft.fft(rho_data - 1))
omega = 2 * np.pi * np.fft.fftfreq(time_array.size, dt)

pl.plot(omega, f_hat)
pl.xlabel(r'$\omega$')
pl.ylabel(r'$|FFT(\max(\rho(x))(t))|$')
pl.savefig('omega.png')
pl.clf()

print('Omega:', omega[int(np.amax(f_hat))])
