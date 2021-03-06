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

# Units: l0, t0, m0, e0, n0, T0, v0, B0, E0
# Independent: n0, T0, m0, e0, k0, eps0, E0, B0
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
m0  = 1. # |m_p| units(m)
e0  = 1. # |e| units(e)
k0  = 1. # |k| units(k)
E0  = 1. # |E| units(E)
eps = 1. # |eps0| units(eps0)
mu  = 1. # |mu0| units(mu0)
c   = 1. # |c| units(c)

# Dimensionality considered in velocity space:
p_dim = 2

# Number of devices(GPUs/Accelerators) on each node:
num_devices = 4

mass               = [1 * m0]
boltzmann_constant = k0
charge             = [0 * e0]

# Velocity, length and time scales:
v0 = velocity_scales.sound_speed(T0, k0, 5/3)
l0 = 0.3 # Box Length
t0 = l0 / v0

# Time parameters:
N_cfl   = 0.95
t_final = 10 * t0

# Switch for solver components:
fields_enabled           = False
source_enabled           = False 
instantaneous_collisions = True 

# File-writing Parameters:
# Set to zero for no file-writing
dt_dump_f       = 1    * t0
dt_dump_moments = 0.01 * t0

# Restart(Set to zero for no-restart):
t_restart = 0 * t0

# Variation of collisional-timescale parameter through phase space:
@af.broadcast
def tau(q1, q2, p1, p2, p3):
    return (0 * t0 * p1**0 * q1**0)
