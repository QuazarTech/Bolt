import numpy as np
import pylab as pl

from lib.physical_system import physical_system
from lib.linear_solver.linear_system import linear_system

import domain
import boundary_conditions
import params
from initialize import intial_conditions

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
                         intial_conditions,\
                         advection_terms,\
                         collision_operator.BGK,\
                         moment_defs
                        )

# Declaring a linear system object which will evolve the defined physical system:
ls = linear_system(system)

# Initializing:
ls.init(params)

# Time parameters:
dt      = 0.001
t_final = 1.0

time_array   = np.arange(0, t_final + dt, dt)
density_data = np.zeros_like(time_array)

for time_index, t0 in enumerate(time_array):
  ls.time_step(dt)
  density_data[time_index] = np.max(ls.compute_moments('density'))

pl.plot(density_data)
pl.savefig('plot.png')