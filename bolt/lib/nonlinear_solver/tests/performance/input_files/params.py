import arrayfire as af

# Can be defined as 'electrostatic', 'user-defined'.
# The initial conditions need to be specified under initialize
# Ensure that the initial conditions specified satisfy
# Maxwell's constraint equations
fields_initialize = 'electrostatic'

# Can be defined as 'electrostatic' and 'fdtd'
fields_solver = 'fdtd'

# Can be defined as 'strang' and 'lie'
time_splitting = 'strang'

# Method in q-space
solver_method_in_q = 'FVM'
riemann_solver = 'lax-friedrichs'
reconstruction_method = 'weno5'

# Constants:
mass_particle      = 1
boltzmann_constant = 1
charge_electron    = 0

p_dim       = 3
num_devices = 4

# Variation of collisional-timescale parameter through phase space:
def tau(q1, q2, p1, p2, p3):
    return (1e-3)