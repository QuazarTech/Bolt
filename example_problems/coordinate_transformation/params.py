import numpy as np
import arrayfire as af

fields_type       = 'None'
fields_initialize = 'fft'
fields_solver     = 'None'

# Solver method:
solver_method_in_q = 'FVM'
solver_method_in_p = 'FVM'

reconstruction_method_in_q = 'minmod'
reconstruction_method_in_p = 'minmod'

riemann_solver_in_q = 'upwind-flux'
riemann_solver_in_p = 'upwind-flux'

# Number of devices(GPUs/Accelerators) on each node:
num_devices = 1

# Constants:
mass   = [1]
charge = [0]
eps    = 1

# Solver Switches:
fields_enabled           = True
source_enabled           = True
instantaneous_collisions = False
hybrid_model_enabled     = False
