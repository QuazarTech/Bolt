import arrayfire as af

fields_type       = 'electrostatic'
fields_initialize = 'fft'
fields_solver     = 'fft'

# Method in q-space
solver_method_in_q = 'FVM'
solver_method_in_p = 'FVM'

reconstruction_method_in_q = 'weno5'
reconstruction_method_in_p = 'weno5'

riemann_solver_in_q = 'upwind-flux'
riemann_solver_in_p = 'upwind-flux'

# Constants:
mass_particle      = 1
boltzmann_constant = 1
charge_electron    = 0

# Dimensionality considered in velocity space:
p_dim = 3

# Number of devices(GPUs/Accelerators) on each node:
num_devices = 4

# Restart(Set to zero for no-restart):
t_restart = 0

# File-writing Parameters:
# Set to zero for no file-writing
dt_dump_f       = 0.1
dt_dump_moments = 0.01 

# Time parameters:
N_cfl   = 0.45
t_final = 10

fields_enabled        = False
source_enabled           = False
instantaneous_collisions = True

# Variation of collisional-timescale parameter through phase space:
@af.broadcast
def tau(q1, q2, p1, p2, p3):
    return(0 * q1**0 * p1**0)
