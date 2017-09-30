import arrayfire as af
import numpy as np
import petsc4py
import sys
import h5py
import pylab as pl

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
dt      = 0.001
t_final = 0.5

time_array = np.arange(0, t_final + dt, dt)

# Initializing Array used in storing the data:
rho_data     = np.zeros_like(time_array)
rho_hat_data = np.zeros_like(time_array)

for time_index, t0 in enumerate(time_array):
    print('Computing For Time =', t0)

    n = ls.compute_moments('density')
    rho_data[time_index]     = af.max(n)
    rho_hat_data[time_index] = af.max(af.abs(af.fft(n-1)))
    ls.RK2_timestep(dt)

f_hat = abs(np.fft.fft(rho_data - np.min(rho_data)))
omega = 2 * np.pi * np.fft.fftfreq(time_array.size, dt)

h5f = h5py.File('data.h5', 'w')
h5f.create_dataset('rho', data = rho_data)
h5f.create_dataset('rho_hat', data = rho_hat_data)
h5f.create_dataset('f_hat', data = f_hat)
h5f.create_dataset('time', data = time_array)
h5f.create_dataset('omega', data = omega)
h5f.close()

print('Omega:', omega[int(np.argmax(f_hat[:int(time_array.size/2)]))])
