import numpy as np
import arrayfire as af

rho_background         = 1
temperature_background = 1

# Can be defined as 'fft', 'electrostatic', 'user-defined'.
# The initial conditions need to be specified under initialize
# Ensure that the initial conditions specified satisfy
# Maxwell's constraint equations
fields_initialize = 'fft'

# Can be defined as 'fft', electrostatic', 'fdtd'
fields_solver = 'fft'

# Dimensionality considered in velocity space:
p_dim = 1

# Method in q-space
solver_method_in_q = 'ASL'
riemann_solver = 'lax-friedrichs'
reconstruction_method = 'minmod'

# Number of devices(GPUs/Accelerators) on each node:
num_devices = 1

# Constants:
mass_particle      = 1
boltzmann_constant = 1
charge_electron    = -1

pert_real = 0.04
pert_imag = 0

k_q1 = 0.3
k_q2 = 0

# Variation of collisional-timescale parameter through phase space:
@af.broadcast
def tau(q1, q2, p1, p2, p3):
    return (0.01 * q1**0 * p1**0)
