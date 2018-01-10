import numpy as np
import arrayfire as af

# Can be defined as 'electrostatic', 'user-defined', 'electrodynamic'.
fields_type       = 'electrostatic'
fields_initialize = 'fft'
fields_solver     = 'fft'

# Method in q-space
solver_method_in_q = 'FVM'
solver_method_in_p = 'FVM'

reconstruction_method_in_q = 'weno5'
reconstruction_method_in_p = 'weno5'

riemann_solver_in_q = 'upwind-flux'
riemann_solver_in_p = 'upwind-flux'

# Dimensionality considered in velocity space:
p_dim = 1

# Number of devices(GPUs/Accelerators) on each node:
num_devices = 1

# Constants:
mass               = af.Array([1, 100], (1, 2))
boltzmann_constant = 1
charge             = af.Array([-1, 1], (1, 2))

# Initial Conditions used in initialize:
rho_background_e = 1
rho_background_i = 1

temperature_background_e = 1
temperature_background_i = 1

# Parameter controlling amplitude of perturbation introduced:
alpha = 0.01

# Time parameters:
N_cfl   = 0.1
t_final = 5

# Switch for solver components:
EM_fields_enabled = True
source_enabled    = True

# Variation of collisional-timescale parameter through phase space:
@af.broadcast
def tau(q1, q2, p1, p2, p3):
    return (np.inf * p1**0 * q1**0)
