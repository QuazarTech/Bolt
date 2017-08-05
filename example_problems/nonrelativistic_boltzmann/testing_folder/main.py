import arrayfire as af
import numpy as np
import pylab as pl

af.info()

from lib.physical_system import physical_system
from lib.nonlinear_solver.nonlinear_solver import nonlinear_solver
from lib.linear_solver.linear_solver import linear_solver

import domain
import boundary_conditions
import params
import initialize

import src.nonrelativistic_boltzmann.advection_terms as advection_terms
import src.nonrelativistic_boltzmann.collision_operator as collision_operator
import src.nonrelativistic_boltzmann.moment_defs as moment_defs

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
system = physical_system(domain,\
                         boundary_conditions,\
                         params,\
                         initialize,\
                         advection_terms,\
                         collision_operator.BGK,\
                         moment_defs
                        )

# Declaring a linear system object which will evolve the defined physical system:
nls = nonlinear_solver(system)
ls  = linear_solver(system)

# Time parameters:
dt      = 0.0005
t_final = 0.1

time_array       = np.arange(0, t_final + dt, dt)
density_data_ls  = np.zeros_like(time_array)
density_data_nls = np.zeros_like(time_array)

for time_index, t0 in enumerate(time_array):
  print('Computing For Time =', t0)
  nls.strang_timestep(dt)
  ls.RK2_step(dt)
  density_data_nls[time_index] = af.max(nls.compute_moments('density'))
  density_data_ls[time_index]  = af.max(ls.compute_moments('density'))

pl.plot(time_array, density_data_ls, '--', color = 'black', label = 'Linear Solver')
pl.plot(time_array, density_data_nls, label = 'Nonlinear Solver')
pl.ylabel(r'$\rho$')
pl.xlabel('Time')
pl.legend()
pl.savefig('plot.png')