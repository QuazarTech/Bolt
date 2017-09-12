import numpy as np
import arrayfire as af

# Can be defined as 'electrostatic', 'user-defined'.
# The initial conditions need to be specified under initialize
# Ensure that the initial conditions specified satisfy
# Maxwell's constraint equations
fields_initialize = 'fft'

# Can be defined as 'electrostatic' and 'fdtd'
fields_solver = 'fft'

# Can be defined as 'strang' and 'lie'
time_splitting = 'strang'

# Dimensionality considered in velocity space:
p_dim = 1

# Number of devices(GPUs/Accelerators) on each node:
num_devices = 1

# Constants:
mass_particle = 1
boltzmann_constant = 1
charge_electron = 0

# Initial Conditions used in initialize:
rho_background = 1
temperature_background = 1

p1_bulk_background = 0
p2_bulk_background = 0
p3_bulk_background = 0

pert_real = 0.01
pert_imag = 0.02

k_q1 = 2 * np.pi
k_q2 = 0

# Variation of collisional-timescale parameter through phase space:
def tau(q1, q2, p1, p2, p3):
    return (0.01 * af.broadcast(lambda a,b:a*b, q1**0,p1**0))