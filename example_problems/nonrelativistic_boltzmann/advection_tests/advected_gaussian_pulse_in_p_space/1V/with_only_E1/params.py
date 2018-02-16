import numpy as np
import arrayfire as af

# Can be defined as 'electrostatic', 'electrodynamic' and 'None'
fields_type       = 'None'
fields_initialize = 'user-defined'
fields_solver     = 'None'

solver_method_in_q = 'FVM'
solver_method_in_p = 'FVM'

reconstruction_method_in_q = 'weno5'
reconstruction_method_in_p = 'piecewise-constant'

riemann_solver_in_q = 'upwind-flux'
riemann_solver_in_p = 'upwind-flux'

# Dimensionality considered in velocity space:
p_dim = 1

# Number of devices(GPUs/Accelerators) on each node:
num_devices = 1

# Constants:
mass               = [1, 1]
boltzmann_constant = 1
charge             = [-10, 10]
eps                = 1

# Initial Conditions used in initialize:
rho_background         = 1
temperature_background = 1

p1_bulk_background = 0

fields_enabled           = True
source_enabled           = False
instantaneous_collisions = False

# Variation of collisional-timescale parameter through phase space:
@af.broadcast
def tau(q1, q2, p1, p2, p3):
    return (np.inf * p1**0 * q1**0)
