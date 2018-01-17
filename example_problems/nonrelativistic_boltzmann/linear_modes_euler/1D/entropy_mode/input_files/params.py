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
p_dim = 1

# Number of devices(GPUs/Accelerators) on each node:
num_devices = 4

# Constants:
mass               = [1]
boltzmann_constant = 1
charge             = [0]

# Initial Conditions used in initialize
# NOTE: Density here is number density
density_background     = 1
temperature_background = 1
v1_bulk_background     = 0
v2_bulk_background     = 0
v3_bulk_background     = 0

k_q1  = 2 * np.pi
gamma = 5 / 3

amplitude = 1e-3

# Time parameters:
N_cfl   = 0.32
t_final = 0.5

# Switch for solver components:
EM_fields_enabled = False
source_enabled    = True

# Variation of collisional-timescale parameter through phase space:
@af.broadcast
def tau(q1, q2, p1, p2, p3):
    return (0 * p1**0 * q1**0)
