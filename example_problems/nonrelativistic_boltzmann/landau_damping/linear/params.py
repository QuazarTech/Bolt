import numpy as np
import arrayfire as af

import bolt.src.nonrelativistic_boltzmann.units.length_scales as length_scales
import bolt.src.nonrelativistic_boltzmann.units.time_scales as time_scales
import bolt.src.nonrelativistic_boltzmann.units.velocity_scales as velocity_scales

# Can be defined as 'electrostatic', 'user-defined', 'electrodynamic'.
fields_type       = 'electrostatic'
fields_initialize = 'fft'
fields_solver     = 'fft'

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
n0  = 1. # |n| units(n)
T0  = 1. # |T| units(T)
m0  = 1. # |m_e| units(m)
e0  = 1. # |e| units(e)
k0  = 1. # |k| units(k)
eps = 1. # |eps0| units(eps0)

v0 = velocity_scales.thermal_speed(T0, m0, k0)
t0 = 1/time_scales.plasma_frequency(n0, e0, m0, eps)
l0 = v0 * t0 # Debye length

# Dimensionality considered in velocity space:
p_dim = 1

# Number of devices(GPUs/Accelerators) on each node:
num_devices = 1

mass               = [1 * m0]
boltzmann_constant = k0
charge             = [-1 * e0]

# Initial Conditions used in initialize:
n_background           = 1 * n0  
temperature_background = 1 * T0

# Parameter controlling amplitude of perturbation introduced:
alpha = 0.01

# Time parameters:
N_cfl   = 0.32
t_final = 20.0

# Switch for solver components:
fields_enabled           = True
source_enabled           = False
instantaneous_collisions = False

# Variation of collisional-timescale parameter through phase space:
@af.broadcast
def tau(q1, q2, p1, p2, p3):
    return (np.inf * t0 * p1**0 * q1**0)
