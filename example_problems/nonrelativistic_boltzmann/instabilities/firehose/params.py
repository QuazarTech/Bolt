import numpy as np
import arrayfire as af

fields_type       = 'electrostatic'
fields_initialize = 'fft'
fields_solver     = 'fft'

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
charge             = [-10]

# Initial Conditions used in initialize:
density_background = 1

beta_1 = 300/np.pi
beta_2 = 3 + 300/np.pi


EM_fields_enabled        = False
source_enabled           = False
instantaneous_collisions = False

# Variation of collisional-timescale parameter through phase space:
@af.broadcast
def tau(q1, q2, p1, p2, p3):
    return (np.inf * p1**0 * q1**0)
