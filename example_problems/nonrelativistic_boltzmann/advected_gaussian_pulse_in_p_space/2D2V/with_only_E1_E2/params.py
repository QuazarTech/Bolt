import numpy as np
import arrayfire as af

# Can be defined as 'electrostatic', 'electrodynamic' and 'user-defined'
fields_type = 'user-defined'

# Can be defined as 'fft', 'snes' and 'user-defined':
fields_initialize = 'user-defined'

solver_method_in_q = 'FVM'
solver_method_in_p = 'FVM'

reconstruction_method_in_q = 'weno5'
reconstruction_method_in_p = 'weno5'

riemann_solver_in_q = 'upwind-flux'
riemann_solver_in_p = 'upwind-flux'

# Dimensionality considered in velocity space:
p_dim = 2

# Number of devices(GPUs/Accelerators) on each node:
num_devices = 1

# Constants:
mass_particle      = 1
boltzmann_constant = 1
charge_electron    = -10

# Initial Conditions used in initialize:
rho_background         = 1
temperature_background = 1

p1_bulk_background = 0
p2_bulk_background = 0

# Variation of collisional-timescale parameter through phase space:
@af.broadcast
def tau(q1, q2, p1, p2, p3):
    return (np.inf * p1**0 * q1**0)

def user_defined_E(q1, q2, t):
    
    E1 = 2 * q1**0
    E2 = 3 * q1**0
    E3 = 0 * q1**0

    return(E1, E2, E3)

def user_defined_B(q1, q2, t):

    B1 = 0. * q1**0
    B2 = 0. * q1**0 
    B3 = 0. * q1**0

    return(B1, B2, B3)
