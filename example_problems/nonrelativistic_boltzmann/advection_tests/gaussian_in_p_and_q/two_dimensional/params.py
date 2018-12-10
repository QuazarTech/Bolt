import numpy as np
import arrayfire as af

fields_type       = 'electrodynamic'
fields_initialize = 'None'
fields_solver     = 'fdtd'

# Solver method:
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
charge             = [0]

# Parameters used in intialization:
q10 = 0.5
q20 = 0.5
p10 = 1
p20 = 1

sigma_q         = 0.05
sigma_p         = 0.4
t_final         = 0.5
dt_dump_moments = 0.001


# Solver Switches:
fields_enabled           = False
source_enabled           = False
instantaneous_collisions = False
hybrid_model_enabled     = False

# Variation of collisional-timescale parameter through phase space:
@af.broadcast
def tau(q1, q2, p1, p2, p3):
    return (np.inf * p1**0 * q1**0)
