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

reconstruction_method_in_q = 'weno5'
reconstruction_method_in_p = 'weno5'

riemann_solver_in_q = 'upwind-flux'
riemann_solver_in_p = 'upwind-flux'

# Units: l0, t0, m0, e0, n0, T0, v0
# Independent: n0, T0, m0, e0, k0, eps0
# Dependent  : l0, t0, v0

# Plasma parameters:
# Number density  ~ 0.1 m^-3
# Temperature     ~ 1000 K
# Mass            ~ m_p kg
# Electric charge ~ 
# Boltzmann const ~ 

# How different are plasmas with 1 K and 1000 K

n0 = 1  # n0 m^-3
T0 = 1  # T0 K
m0 = 1  # m_p kg
e0 = 1  # e C
k0 = 1  # k J/K
eps0 = 1 # eps0 farads/m

l0 = length_scales.debye_length(n0, T0, e0, eps0, k0)
v0 = velocity_scales.thermal_speed(T0, m0, k0)
t0 = 1/time_scales.plasma_freq(n0, e0, eps0, m0)

# Dimensionality considered in velocity space:
p_dim = 1

# Number of devices(GPUs/Accelerators) on each node:
num_devices = 1

# Constants:
m_e = 1/100 * m0
m_i = 1     * m0
mass               = [m_e, m_i] # m_e, m_i
boltzmann_constant = k0
charge             = [-1 * e0, 1 * e0] # e_e, e_i

# e_e*/m_e*E*df_e/dv, e_i/m_i*E*df_i/dv

# Initial Conditions used in initialize:
n_background_e = 1 * n0  
n_background_i = 1 * n0

temperature_background_e = 2.5 * T0
temperature_background_i = 1   * T0

# Parameter controlling amplitude of perturbation introduced:
alpha = 0.01

# Time parameters:
N_cfl   = 0.9
t_final = 6 * t0

# Switch for solver components:
fields_enabled        = True
source_enabled           = False
instantaneous_collisions = False

# Variation of collisional-timescale parameter through phase space:
@af.broadcast
def tau(q1, q2, p1, p2, p3):
    return (np.inf * t0 * p1**0 * q1**0)
