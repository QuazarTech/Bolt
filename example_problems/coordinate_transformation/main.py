import arrayfire as af
import numpy as np
import h5py

from bolt.lib.physical_system import physical_system
from bolt.lib.nonlinear.nonlinear_solver import nonlinear_solver

import domain
import boundary_conditions
import params
import initialize

import bolt.src.coordinate_transformation.advection_terms as advection_terms
import bolt.src.coordinate_transformation.source_term as source_term
import bolt.src.coordinate_transformation.moments as moments

# Defining the physical system to be solved:
system = physical_system(domain,
                         boundary_conditions,
                         params,
                         initialize,
                         advection_terms,
                         source_term.source_term,
                         moments
                        )

# Declaring a linear system object which will evolve the defined physical system:
nls = nonlinear_solver(system)
N_g = nls.N_ghost

# Time parameters:
dt      = 0.00025
t_final = 1.0

time_array = np.arange(dt, t_final + dt, dt)

n_nls = nls.compute_moments('density')

f_initial = nls.f
nls.dump_moments('dump/0000')

for time_index, t0 in enumerate(time_array):

    print('Time = %.3f'%t0)
    
    nls.strang_timestep(dt)
    nls.dump_moments('dump/%04d'%(time_index+1))
