import numpy as np
import arrayfire as af

fields_type       = 'electrostatic'
fields_initialize = 'fft'
fields_solver     = 'fft'

# Dimensionality considered in velocity space:
p_dim = 1

# Method in q-space
solver_method_in_q = 'FVM'
solver_method_in_p = 'FVM'

riemann_solver_in_q = 'upwind-flux'
riemann_solver_in_p = 'upwind-flux'

reconstruction_method_in_q = 'minmod'
reconstruction_method_in_p = 'minmod'

t_final = 50
N_cfl   = 0.04

# Number of devices(GPUs/Accelerators) on each node:
num_devices = 1

# Constants:
mass               = [1]
boltzmann_constant = 1
charge             = [-1]
eps                = 1

k_q1  = 0.5
alpha = 0.01

fields_enabled           = True
source_enabled           = False
instantaneous_collisions = False
hybrid_model_enabled     = False

# Variation of collisional-timescale parameter through phase space:
@af.broadcast
def tau(q1, q2, p1, p2, p3):
    return (0.001 * q1**0 * p1**0)
