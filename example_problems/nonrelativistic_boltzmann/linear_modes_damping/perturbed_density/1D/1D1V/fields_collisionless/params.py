import numpy as np
import arrayfire as af

# Can be defined as 'electrostatic', 'user-defined', 'electrodynamic'.
fields_type       = 'electrodynamic'
fields_initialize = 'fft'
fields_solver     = 'fdtd'

# Method in q-space
solver_method_in_q = 'FVM'
solver_method_in_p = 'FVM'

reconstruction_method_in_q = 'weno5'
reconstruction_method_in_p = 'weno5'

riemann_solver_in_q = 'upwind-flux'
riemann_solver_in_p = 'upwind-flux'

# Dimensionality considered in velocity space:
p_dim = 1

# Number of devices(GPUs/Accelerators) on each node:
num_devices = 1

# Constants:
mass               = af.Array([1, 1], (1, 2))
boltzmann_constant = 1
charge             = af.Array([-10, 5], (1, 2))

# Initial Conditions used in initialize:
rho_background_species_1 = 1
rho_background_species_2 = 2

temperature_background_species_1 = 1
temperature_background_species_2 = 1

v1_bulk_background_species_1 = 0
v1_bulk_background_species_2 = 0

pert_real_species_1 = 0.01
pert_real_species_2 = 0.01

pert_imag_species_1 = 0.02
pert_imag_species_2 = 0.02

k_q1_species_1 = 2 * np.pi
k_q1_species_2 = 2 * np.pi

k_q2_species_1 = 0 * np.pi
k_q2_species_2 = 0 * np.pi

# Time parameters:
N_cfl   = 0.32
t_final = 0.5

# Switch for solver components:
EM_fields_enabled = True
source_enabled    = True

# Variation of collisional-timescale parameter through phase space:
@af.broadcast
def tau(q1, q2, p1, p2, p3):
    return (0.01 * p1**0 * q1**0)
