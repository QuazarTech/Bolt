import numpy as np
import arrayfire as af

import bolt.src.nonrelativistic_boltzmann.units.length_scales as length_scales
import bolt.src.nonrelativistic_boltzmann.units.time_scales as time_scales
import bolt.src.nonrelativistic_boltzmann.units.velocity_scales as velocity_scales

# Can be defined as 'electrostatic', 'user-defined', 'electrodynamic'.
fields_type       = 'electrodynamic'
fields_initialize = 'user-defined'
fields_solver     = 'fdtd'

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
mu  = 1. # |mu0| units(mu0)

# Dimensionality considered in velocity space:
p_dim = 3

# Number of devices(GPUs/Accelerators) on each node:
num_devices = 4

# Mass of electron and ion:
m_e = (1 / 100) * m0
m_i = 1         * m0

# Charge of electron and ion:
e_e = -1 * e0
e_i =  1 * e0

mass               = [m_e, m_i]
boltzmann_constant = k0
charge             = [e_e, e_i]

# Boundary conditions for the density and temperature of left zone, 
# Setup as initial conditions throughout domain:
n_left = 1 * n0
T_left = 1 * T0

plasma_beta = 100 # β = p / (B^2 / 2μ)
# Setting magnetic field along x using plasma beta:
B1 = np.sqrt(2 * mu * n_left * T_left / plasma_beta)

# Velocity, length and time scales:
t0 = 1 / time_scales.cyclotron_frequency(B1, e_i, m_i)
v0 = velocity_scales.alfven_velocity(B1, n_left, m_i, mu)
l0 = v0 * t0 # ion skin depth

# Setting permeability:
c   = 300 * v0 # |c| units(c)
eps = 1 / (c**2 * mu)

# Setting bulk velocity of left boundary:
# Also setup as initial conditions throughout domain:
v1_bulk_left = 1 * v0

# Time parameters:
N_cfl   = 0.1
t_final = 200 * t0

# Switch for solver components:
fields_enabled           = True
source_enabled           = False
instantaneous_collisions = False

# File-writing Parameters:
# Set to zero for no file-writing
dt_dump_f       = 1 * t0
# ALWAYS set dump moments and dump fields at same frequency:
dt_dump_moments = dt_dump_fields = 0.001 * t0

# Restart(Set to zero for no-restart):
t_restart = 0 * t0

# Variation of collisional-timescale parameter through phase space:
@af.broadcast
def tau(q1, q2, p1, p2, p3):
    return (np.inf * t0 * p1**0 * q1**0)
