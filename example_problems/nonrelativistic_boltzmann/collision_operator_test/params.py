import numpy as np
import arrayfire as af

fields_type       = 'electrodynamic'
fields_initialize = 'fft'
fields_solver     = 'fdtd'

solver_method_in_q = 'FVM'
solver_method_in_p = 'FVM'

reconstruction_method_in_q = 'minmod'
reconstruction_method_in_p = 'minmod'

riemann_solver_in_q = 'upwind-flux'
riemann_solver_in_p = 'upwind-flux'

instantaneous_collisions = 0

# Dimensionality considered in velocity space:
p_dim = 1

# Number of devices(GPUs/Accelerators) on each node:
num_devices = 1

# Constants:
mass               = [1]
boltzmann_constant = 1
charge             = [0]

fields_enabled           = False
source_enabled           = True
instantaneous_collisions = False
hybrid_model_enabled     = False

# Variation of collisional-timescale parameter through phase space:
@af.broadcast
def tau(q1, q2, p1, p2, p3):
    return (0.01 * p1**0 * q1**0)
