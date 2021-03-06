import arrayfire as af
import numpy as np
import h5py 

from bolt.lib.physical_system import physical_system
from bolt.lib.nonlinear.nonlinear_solver import nonlinear_solver

import domain
import boundary_conditions
import params
import initialize

import bolt.src.nonrelativistic_boltzmann.advection_terms as advection_terms
import bolt.src.nonrelativistic_boltzmann.collision_operator as collision_operator
import bolt.src.nonrelativistic_boltzmann.moments as moments


# Defining the physical system to be solved:
system = physical_system(domain,
                         boundary_conditions,
                         params,
                         initialize,
                         advection_terms,
                         collision_operator.BGK,
                         moments
                        )

# Declaring a linear system object which will evolve the defined physical system:
nls = nonlinear_solver(system)

# Time parameters:
dt      = 0.0025
t_final = 2.5

time_array = np.arange(dt, t_final + dt, dt)

n_nls = nls.compute_moments('density')

h5f = h5py.File('dump/0000.h5', 'w')
h5f.create_dataset('q1', data = nls.q1_center)
h5f.create_dataset('q2', data = nls.q2_center)
h5f.create_dataset('n', data = n_nls)
h5f.close()

f_initial = nls.f.copy()

for time_index, t0 in enumerate(time_array):
    
    print('Computing for Time =', t0)

    nls.strang_timestep(dt)
    n_nls = nls.compute_moments('density')
    
    h5f = h5py.File('dump/%04d'%(time_index+1) + '.h5', 'w')
    h5f.create_dataset('n', data = n_nls)
    h5f.close()

print(af.mean(af.abs(nls.f[:, :, 3:-3, 3:-3] - f_initial[:, :, 3:-3, 3:-3])))
