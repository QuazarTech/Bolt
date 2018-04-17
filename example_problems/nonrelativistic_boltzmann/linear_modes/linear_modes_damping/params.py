import numpy as np
import arrayfire as af

import bolt.src.nonrelativistic_boltzmann.units.length_scales as length_scales
import bolt.src.nonrelativistic_boltzmann.units.time_scales as time_scales
import bolt.src.nonrelativistic_boltzmann.units.velocity_scales as velocity_scales

# Can be defined as 'electrostatic', 'electrodynamic', 'None'
fields_type       = 'electrodynamic'
fields_initialize = 'fft'
fields_solver     = 'fdtd'

# Method in q-space
solver_method_in_q = 'FVM'
solver_method_in_p = 'FVM'

reconstruction_method_in_q = 'minmod'
reconstruction_method_in_p = 'minmod'

riemann_solver_in_q = 'upwind-flux'
riemann_solver_in_p = 'upwind-flux'

# Units: l0, t0, m0, e0, n0, T0, v0
# Independent: n0, T0, m0, e0, k0, eps0
# Dependent  : l0, t0, v0

# Plasma parameters (given):
# Number density  ~ n; n = |n| units(n)
# Temperature     ~ T; T = |T| units(T)

# Fundamental consts: 
# Mass            ~ m_p; m_p = |m_p| units(m_p)
# Electric charge ~ e;   e   = |e|   units(e)
# Boltzmann const ~ k;   k   = |k|   units(k)
# Vacuum perm     ~ eps0; eps0 = |eps0| units(eps0)

# Now choosing units: 
n0  = 1 # |n| units(n)
T0  = 1 # |T| units(T)
m0  = 1 # |m_p| units(m)
e0  = 1 # |e| units(e)
k0  = 1 # |k| units(k)
eps = 1 # |eps0| units(eps0)
mu  = 1 # |mu0| units(mu0)

l0 = length_scales.debye_length(n0, T0, e0, k0, eps)
t0 = 1 / time_scales.plasma_frequency(n0, e0, m0, eps)
v0 = l0 / t0

# Dimensionality considered in velocity space:
p_dim = 1

# Number of devices(GPUs/Accelerators) on each node:
num_devices = 1

# Constants:
m_e = 1 * m0

mass               = [m_e]
boltzmann_constant = k0
charge             = [10 * e0]

# Initial Conditions used in initialize:
density_background     = 1 * n0  
temperature_background = 1 * T0
v1_bulk_background     = 0 * v0
v2_bulk_background     = 0 * v0
v3_bulk_background     = 0 * v0

# Used in hybrid model:
fluid_electron_temperature = 1 * T0

# Wavenumber of perturbation
k_q1 = 2 * np.pi 
k_q2 = 0

# Parameter controlling amplitude of perturbation introduced:
pert_real = 0.01
pert_imag = 0

# Time parameters:
N_cfl   = 0.32
t_final = 1.0 * t0

# Switch for solver components:
fields_enabled           = True
source_enabled           = False
instantaneous_collisions = False
hybrid_model_enabled     = True

# Variation of collisional-timescale parameter through phase space:
@af.broadcast
def tau(q1, q2, p1, p2, p3):
    return (np.inf * t0 * p1**0 * q1**0)
