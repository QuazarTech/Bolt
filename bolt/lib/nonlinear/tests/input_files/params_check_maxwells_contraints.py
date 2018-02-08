import numpy as np
import arrayfire as af

fields_type       = 'electrodynamic'
fields_initialize = 'fft + user-defined magnetic fields'
fields_solver     = 'fdtd'

# Method in q-space
solver_method_in_q = 'FVM'
solver_method_in_p = 'FVM'

reconstruction_method_in_q = 'weno5'
reconstruction_method_in_p = 'weno5'

riemann_solver_in_q = 'upwind-flux'
riemann_solver_in_p = 'upwind-flux'

# Constants:
mass               = [1]
boltzmann_constant = 1
charge             = [0]
eps                = 1

fields_enabled = True

pert_real = 1e-5
pert_imag = 2e-5

k_q1 = 2 * np.pi
k_q2 = 4 * np.pi

num_devices = 1
