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

# Dimensionality considered in velocity space:
p_dim = 1

# Number of devices(GPUs/Accelerators) on each node:
num_devices = 1

# Constants:
mass               = [1]
boltzmann_constant = 1
charge             = [-10]

fields_enabled           = True
source_enabled           = False
instantaneous_collisions = False
hybrid_model_enabled     = False

# Variation of collisional-timescale parameter through phase space:
@af.broadcast
def tau(q1, q2, p1, p2, p3):
    return (0.01 * p1**0 * q1**0)

# Initial Conditions used in initialize:
# Here density refers to number density
density_background     = 1
temperature_background = 1

p1_bulk_background = 0

pert_real = 0.01
pert_imag = 0.02

eps = 1
mu  = 1

k_q1 = 2 * np.pi
