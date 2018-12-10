import numpy as np
import arrayfire as af

# Can be defined as 'electrostatic', 'user-defined', 'electrodynamic'.
fields_type       = 'electrodynamic'
fields_initialize = 'fft'
fields_solver     = 'fdtd'

# Method in q-space
solver_method_in_q = 'FVM'
solver_method_in_p = 'FVM'

reconstruction_method_in_q = 'minmod'
reconstruction_method_in_p = 'minmod'

riemann_solver_in_q = 'upwind-flux'
riemann_solver_in_p = 'upwind-flux'

# Dimensionality considered in velocity space:
p_dim = 1

# Number of devices(GPUs/Accelerators) on each node:
num_devices = 1

# Constants:
mass               = [1]
boltzmann_constant = 1
charge             = [0]

# Initial Conditions used in initialize
# NOTE: Density here is number density
density_background     = 1
temperature_background = 1
v1_bulk_background     = 0

k_q1      = 2 * np.pi
amplitude = 1e-3

# Time parameters:
N_cfl   = 32e-6
t_final = 0.1

# Switch for solver components:
fields_enabled           = False
source_enabled           = True
instantaneous_collisions = False
hybrid_model_enabled     = False

# Variation of collisional-timescale parameter through phase space:
@af.broadcast
def tau(q1, q2, p1, p2, p3):
    return (1e-7 * p1**0 * q1**0)
