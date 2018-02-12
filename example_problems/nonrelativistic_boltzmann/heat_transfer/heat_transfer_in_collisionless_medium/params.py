import numpy as np
import arrayfire as af

fields_type       = 'electrodynamic'
fields_initialize = 'fft + user-defined magnetic fields'
fields_solver     = 'fdtd'

solver_method_in_q = 'FVM'
solver_method_in_p = 'FVM'

reconstruction_method_in_q = 'weno5'
reconstruction_method_in_p = 'weno5'

riemann_solver_in_q = 'upwind-flux'
riemann_solver_in_p = 'upwind-flux'

instantaneous_collisions = 0

# Dimensionality considered in velocity space:
p_dim = 1

# Number of devices(GPUs/Accelerators) on each node:
num_devices = 4

# Constants:
mass               = [1]
boltzmann_constant = 1
charge             = [-1]
eps                = 1
mu                 = 1

# Initial Conditions used in initialize:
rho_background         = 1
temperature_background = 1

p1_bulk_background = 0
p2_bulk_background = 0
p3_bulk_background = 0

pert_real = 1e-5
pert_imag = 2e-5

k_q1 = 2 * np.pi
k_q2 = 4 * np.pi

fields_enabled           = True
source_enabled           = True
instantaneous_collisions = False

# Variation of collisional-timescale parameter through phase space:
@af.broadcast
def tau(q1, q2, p1, p2, p3):
    return (np.inf * p1**0 * q1**0)
