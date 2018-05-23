import arrayfire as af

fields_type       = 'electrodynamic'
fields_initialize = 'user-defined'
fields_solver     = 'fdtd'

# Method in q-space
solver_method_in_q = 'FVM'
solver_method_in_p = 'FVM'

reconstruction_method_in_q = 'minmod'
reconstruction_method_in_p = 'minmod'

riemann_solver_in_q = 'upwind-flux'
riemann_solver_in_p = 'upwind-flux'

# Constants:
mass               = [1]
boltzmann_constant = 1
charge             = [1]

eps = 1
mu  = 1

fields_enabled           = True
source_enabled           = False
instantaneous_collisions = False
hybrid_model_enabled     = False

num_devices = 4
