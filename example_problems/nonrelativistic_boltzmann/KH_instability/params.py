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

# Constants:
mass_particle = 1
boltzmann_constant = 1
charge_electron = 0

p_dim = 3

# Variation of collisional-timescale parameter through phase space:
def tau(q1, q2, p1, p2, p3):
    return (0.01 * af.broadcast(lambda a, b:a*b, q1**0,p1**0))
