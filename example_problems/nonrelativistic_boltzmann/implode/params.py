import numpy as np
import arrayfire as af

# Can be defined as 'electrostatic', 'user-defined'.
# The initial conditions need to be specified under initialize
# Ensure that the initial conditions specified satisfy
# Maxwell's constraint equations
fields_initialize = 'fft'

# Can be defined as 'electrostatic' and 'fdtd'
fields_solver = 'fdtd'

# Method in q-space
solver_method_in_q = 'FVM'
solver_method_in_p = 'FVM'

reconstruction_method_in_q = 'weno5'
reconstruction_method_in_p = 'weno5'

riemann_solver = 'upwind-flux'

# Dimensionality considered in velocity space:
p_dim = 3

# Number of devices(GPUs/Accelerators) on each node:
num_devices = 4

# Constants:
mass_particle      = 1
boltzmann_constant = 1
charge_electron    = 0

# Restart(Set to zero for no-restart):
t_restart = 0

# File-writing Parameters:
# Set to zero for no file-writing
dt_dump_f       = 0.1
dt_dump_moments = 0.01 

# Time parameters:
N_cfl   = 0.96
t_final = 2.5

# Variation of collisional-timescale parameter through phase space:
@af.broadcast
def tau(q1, q2, p1, p2, p3):
    return (0 * q1**0 * p1**0)
