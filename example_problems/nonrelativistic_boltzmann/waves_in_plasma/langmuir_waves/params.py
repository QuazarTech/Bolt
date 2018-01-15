import numpy as np
import arrayfire as af

# Can be defined as 'electrostatic', 'user-defined', 'electrodynamic'.
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

# Dimensionality considered in velocity space:
p_dim = 1

# Number of devices(GPUs/Accelerators) on each node:
num_devices = 1

# Constants:
#kT0 = 1
#m0  = 1
#e0  = 1
#v0  = (kT0/m0)**0.5
#plasma_freq = (n * e0**2/m0)
#t0  = 1/plasma_freq

mass_ion      = 1     #* m0
mass_electron = 1/100 #* m0
mass               = af.Array([mass_electron, mass_ion], (1, 2))
boltzmann_constant = 1
charge             = af.Array([-1, 1], (1, 2)) #e0

# Initial Conditions used in initialize:
rho_background_e = 1 # 1/l0^3; l0 = v0 t0; v0 = (2kT0/m0)^0.5
rho_background_i = 1 # 1/l0^3

temperature_background_i = 1 # kT_i
temperature_background_e = 1 # kT_i

# Parameter controlling amplitude of perturbation introduced:
alpha = 0.01

# Time parameters:
N_cfl   = 0.1
t_final = 5 # t0 = 1/omega_pi

# Switch for solver components:
EM_fields_enabled = True
source_enabled    = True

# Variation of collisional-timescale parameter through phase space:
@af.broadcast
def tau(q1, q2, p1, p2, p3):
    return (np.inf * p1**0 * q1**0)
