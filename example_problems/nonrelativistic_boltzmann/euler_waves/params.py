import numpy as np
import arrayfire as af

# Can be defined as 'electrostatic', 'user-defined', 'electrodynamic'.
fields_type       = 'electrodynamic'
fields_initialize = 'fft'
fields_solver     = 'fdtd'

# Method in q-space
solver_method_in_q = 'FVM'
solver_method_in_p = 'FVM'

reconstruction_method_in_q = 'weno5'
reconstruction_method_in_p = 'weno5'

riemann_solver_in_q = 'upwind-flux'
riemann_solver_in_p = 'upwind-flux'

# Dimensionality considered in velocity space:
p_dim = 3

# Number of devices(GPUs/Accelerators) on each node:
num_devices = 1

# Constants:
mass               = af.Array([1])
boltzmann_constant = 1
charge             = af.Array([0])

# Initial Conditions used in initialize:
rho_background         = 1
temperature_background = 1
v1_bulk_background     = 0
v2_bulk_background     = 0
v3_bulk_background     = 0

k_q1  = 2 * np.pi
gamma = 5 / 3

# Introducing the perturbation amounts:
# This is obtained from the Sage Worksheet(https://goo.gl/Sh8Nqt):
# Plugging in the value from the Eigenvectors:
pert_rho = 1
pert_v1  = np.sqrt(gamma * temperature_background) / rho_background
pert_T   = temperature_background * (gamma - 1) / rho_background

# Plugging in the Eigenvalue:
# This is used in the analytic solution:
omega = np.sqrt(temperature_background * gamma)* k_q1 * 1j

# Time parameters:
N_cfl   = 0.1
t_final = 0.5

# Switch for solver components:
EM_fields_enabled = False
source_enabled    = True

# Variation of collisional-timescale parameter through phase space:
@af.broadcast
def tau(q1, q2, p1, p2, p3):
    return (0 * p1**0 * q1**0)
