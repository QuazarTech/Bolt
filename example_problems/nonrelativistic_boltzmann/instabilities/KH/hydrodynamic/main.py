import arrayfire as af
import numpy as np
from petsc4py import PETSc

from tqdm import trange

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

# Defining the physical system to be solved:
system = physical_system(domain,
                         boundary_conditions,
                         params,
                         initialize,
                         advection_terms,
                         collision_operator.BGK,
                         moment_defs
                         )

# Declaring a linear system object which will evolve
# the defined physical system:
nls = nonlinear_solver(system)
# Printing device, and backend details:
af.info()

density_vec = nls._da_ksp.createGlobalVec()
density_vec_value = nls._da_ksp.getVecArray(density_vec)

PETSc.Object.setName(density_vec, 'density')

# Time parameters:
dt      = 0.00001
t_final = 0.001

time_array = np.arange(0, t_final + dt, dt)

def time_evolution():

    for time_index in trange(time_array.size):
        nls.strang_timestep(dt)

        # density = nls.compute_moments('density')

        # if(time_index%1000==0):
        #     density_vec_value[:] = np.array(density[3:-3, 3:-3])
        #     viewer = PETSc.Viewer().createHDF5('dump/density_' + str(time_index) + '.h5',
        #                                        'w', comm=nls._comm)
        #     viewer(density_vec)

time_evolution()
