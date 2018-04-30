import numpy as np
import arrayfire as af

fields_type       = 'user-defined'
fields_initialize = 'user-defined'
fields_solver     = 'None'

solver_method_in_q = 'FVM'
solver_method_in_p = 'FVM'

reconstruction_method_in_q = 'minmod'
reconstruction_method_in_p = 'minmod'

riemann_solver_in_q = 'upwind-flux'
riemann_solver_in_p = 'upwind-flux'

# Dimensionality considered in velocity space:
p_dim = 2

# Number of devices(GPUs/Accelerators) on each node:
num_devices = 1

# Constants:
mass               = [1]
boltzmann_constant = 1
charge             = [1]
eps                = 1

# Initial Conditions used in initialize:
rho_background         = 1
temperature_background = 1

v1_bulk_background = 2
v2_bulk_background = 2

# Switch for solver components:
fields_enabled           = True
source_enabled           = False
instantaneous_collisions = False
hybrid_model_enabled     = False

# Variation of collisional-timescale parameter through phase space:
@af.broadcast
def tau(q1, q2, p1, p2, p3):
    return (np.inf * p1**0 * q1**0)
