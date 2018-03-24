import numpy as np
import arrayfire as af
from petsc4py import PETSc

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
mu  = 1. # |mu0| units(mu0)

# Printing Details About the Different Scales:
PETSc.Sys.Print("==================================================")
PETSc.Sys.Print("             Independent Units Chosen             ")
PETSc.Sys.Print("==================================================")
PETSc.Sys.Print("Density              :", n0, "|n| units(n)")
PETSc.Sys.Print("Temperature          :", T0, "|T| units(n)")
PETSc.Sys.Print("Mass                 :", m0, "|m_e| units(m)")
PETSc.Sys.Print("Charge               :", e0, "|e| units(e)")
PETSc.Sys.Print("Boltzmann Constant   :", k0, "|k| units(k)")
PETSc.Sys.Print("Magnetic Permeability:", mu, "|mu0| units(mu0)")
PETSc.Sys.Print("==================================================\n")

# Dimensionality considered in velocity space:
p_dim = 2
# p_dim sets the adiabatic constant gamma:
gamma = 2

# Number of devices(GPUs/Accelerators) on each node:
num_devices = 4

# Mass of electron and positron:
m_e = 1 * m0
m_p = 1 * m0

# Charge of electron and positron:
e_e = -1 * e0
e_p =  1 * e0

mass               = [m_e, m_p]
boltzmann_constant = k0
charge             = [e_e, e_p]

# Background Quantities:
density_background     = 1 * n0
temperature_background = 1 * T0

plasma_beta = 100 # β = p / (B^2 / 2μ)
# Setting magnetic field along x using plasma beta:
B1 = np.sqrt(2 * mu * density_background * temperature_background / plasma_beta)

# Velocity, length and time scales:
t0 = 1 / time_scales.cyclotron_frequency(B1, e0, m0)
v0 = velocity_scales.alfven_velocity(B1, density_background, m0, mu)
l0 = v0 * t0 # positron skin depth

# Setting the length of the box:
L_x = L_y = 10 * l0

# Setting Maximum Velocity of Phase Space Grid:
v_max = 30 * v0

# Calculating Permittivity:
c   = v_max
eps = 1 / (c**2 * mu)

# Velocity Scales:
thermal_speed   = velocity_scales.thermal_speed(temperature_background, m0, k0)
sound_speed     = velocity_scales.sound_speed(temperature_background, k0, gamma)
alfven_velocity = velocity_scales.alfven_velocity(B1, density_background, m0, mu) 

# Length scales:
debye_length = length_scales.debye_length(density_background, temperature_background, e0, k0, eps)
skin_depth   = length_scales.skin_depth(density_background, e0, c, m0, eps)
gyroradius   = length_scales.gyroradius(velocity_scales.thermal_speed(temperature_background, m0, k0), B1, e0, m0)

# Time scales:
plasma_frequency     = time_scales.plasma_frequency(density_background, e0, m0, eps)
cyclotron_frequency  = time_scales.cyclotron_frequency(B1, e0, m0)
alfven_crossing_time = time_scales.alfven_crossing_time(min(L_x, L_y), B1, density_background, m0, mu)
sound_crossing_time  = time_scales.sound_crossing_time(min(L_x, L_y), temperature_background, k0, gamma)

# Time parameters:
N_cfl   = 0.4
t_final = 20 * t0

PETSc.Sys.Print("==================================================")
PETSc.Sys.Print("          Length Scales of the System             ")
PETSc.Sys.Print("==================================================")
PETSc.Sys.Print("Debye Length       :", debye_length)
PETSc.Sys.Print("Skin Depth         :", skin_depth)
PETSc.Sys.Print("Gyroradius         :", gyroradius)
PETSc.Sys.Print("Chosen Length Scale:", l0)
PETSc.Sys.Print("Length_x           :", L_x / l0, "|l0| units(l0)")
PETSc.Sys.Print("Length_y           :", L_y / l0, "|l0| units(l0)")
PETSc.Sys.Print("==================================================\n")

PETSc.Sys.Print("==================================================")
PETSc.Sys.Print("           Time Scales of the System              ")
PETSc.Sys.Print("==================================================")
PETSc.Sys.Print("Plasma Frequency    :", plasma_frequency)
PETSc.Sys.Print("Cyclotron Frequency :", cyclotron_frequency)
PETSc.Sys.Print("Alfven Crossing Time:", alfven_crossing_time)
PETSc.Sys.Print("Sound Crossing Time :", sound_crossing_time)
PETSc.Sys.Print("Chosen Time Scale   :", t0)
PETSc.Sys.Print("Final Time          :", t_final / t0, "|t0| units(t0)")
PETSc.Sys.Print("==================================================\n")

PETSc.Sys.Print("==================================================")
PETSc.Sys.Print("          Velocity Scales of the System           ")
PETSc.Sys.Print("==================================================")
PETSc.Sys.Print("Thermal Speed   :", thermal_speed)
PETSc.Sys.Print("Sound Speed     :", sound_speed)
PETSc.Sys.Print("Alfven Velocity :", alfven_velocity)
PETSc.Sys.Print("Velocity        :", v0)
PETSc.Sys.Print("Maximum Velocity:", v_max / v0, "|v0| units(v0)")
PETSc.Sys.Print("==================================================\n")

PETSc.Sys.Print("==================================================")
PETSc.Sys.Print("              Other Dependent Units               ")
PETSc.Sys.Print("==================================================")
PETSc.Sys.Print("Light Speed :", c / v0, "|v0| units(v0)")
PETSc.Sys.Print("Permittivity:", eps)
PETSc.Sys.Print("==================================================\n")

v1_bulk_electron =  1 * v0
v1_bulk_positron = -1 * v0

v2_bulk_electron =  1 * v0
v2_bulk_positron = -1 * v0

# Switch for solver components:
fields_enabled           = True
source_enabled           = False
instantaneous_collisions = False

# File-writing Parameters:
# Set to zero for no file-writing
dt_dump_f       = 1 * t0
# ALWAYS set dump moments and dump fields at same frequency:
dt_dump_moments = dt_dump_fields = 0.01 * t0

# Restart(Set to zero for no-restart):
t_restart = 0 * t0

# Variation of collisional-timescale parameter through phase space:
@af.broadcast
def tau(q1, q2, p1, p2, p3):
    return (np.inf * t0 * p1**0 * q1**0)
