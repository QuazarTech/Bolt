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
solver_method_in_q = 'ASL'
solver_method_in_p = 'ASL'

reconstruction_method_in_q = 'weno5'
reconstruction_method_in_p = 'weno5'

riemann_solver = 'upwind-flux'

# Dimensionality considered in velocity space:
p_dim = 2

# Number of devices(GPUs/Accelerators) on each node:
num_devices = 1

# Constants:
mass_particle      = 1
boltzmann_constant = 1
charge_electron    = 0

# Initial Conditions used in initialize:
rho_background         = 1
temperature_background = 1

p1_bulk_background = 0
p2_bulk_background = 0

pert_real = 0.01
pert_imag = 0.02

k_q1 = 2 * np.pi
k_q2 = 4 * np.pi

# Variation of collisional-timescale parameter through phase space:
@af.broadcast
def tau(q1, q2, p1, p2, p3):
    return (np.inf * p1**0 * q1**0)
