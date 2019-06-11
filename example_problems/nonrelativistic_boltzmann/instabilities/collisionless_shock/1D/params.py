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

# Alternate set of units to consider:
# Temperature isn't a very good base unit
# Set v0 = 1(that is velocity is a base unit)
# Set T0 = 1/2 m0 v0^2 / k_B, and set T = eps * T0;eps = small number
# B0 is set the same way as before; B0 = sqrt(n_background * m0) * u_b

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
m0  = 1. # |m_e| units(m)
e0  = 1. # |e| units(e)
k0  = 1. # |k| units(k)
mu  = 1. # |mu0| units(mu0)

# Boundary conditions for the density and temperature of left zone, 
# Setup as initial conditions throughout domain:
n_background = 1    * n0
T_background = 1e-7 * T0

# Bulk velocity for electron is set to be 5e-5. 
# It is later determined this value in terms of v0
# We intend to set the velocities such that electrons and ions both 
# have same initial energy per particle:
# Setting bulk velocity:
u_be = 5e-4
u_bi = 5e-5

# Getting scale B0 by setting omega_c = omega_p * u_b / c
# => B0 = sqrt(n * m / eps) * u_b / c
# => B0 = sqrt(n * m * c**2 * mu) * u_b / c
# => B0 = sqrt(n * m * mu) * u_b
B0 = np.sqrt(n_background * m0) * u_be
B1 = 1e-5 * B0

# Printing Details About the Different Scales:
PETSc.Sys.Print("==================================================")
PETSc.Sys.Print("             Independent Units Chosen             ")
PETSc.Sys.Print("==================================================")
PETSc.Sys.Print("Density              :", n0, "|n| units(n)")
PETSc.Sys.Print("Temperature          :", T0, "|T| units(T)")
PETSc.Sys.Print("Mass                 :", m0, "|m_e| units(m)")
PETSc.Sys.Print("Charge               :", e0, "|e| units(e)")
PETSc.Sys.Print("Boltzmann Constant   :", k0, "|k| units(k)")
PETSc.Sys.Print("Magnetic Permeability:", mu, "|mu0| units(mu0)")
PETSc.Sys.Print("==================================================")
PETSc.Sys.Print("Setting the magnetic field scale                  ")
PETSc.Sys.Print("==================================================")
PETSc.Sys.Print("Magnetic Field       :", B0, "|B| units(B)")
PETSc.Sys.Print("==================================================\n")

# Dimensionality considered in velocity space:
p_dim = 3
gamma = 5 / 3 # adiabatic factor

# Number of devices(GPUs/Accelerators) on each node:
num_devices = 4

# Mass of electron and ion:
m_e = 1   * m0
m_i = 100 * m0

# Charge of electron and ion:
e_e = -1 * e0
e_i =  1 * e0

mass               = [m_e, m_i]
boltzmann_constant = k0
charge             = [e_e, e_i]

# Velocity, length and time scales:
t0 = 1 / time_scales.cyclotron_frequency(B0, e0, m0)
v0 = velocity_scales.alfven_velocity(B0, n_background, m0, mu)
l0 = v0 * t0 # ion skin depth

# Setting lengths of the domain:
L_x = 5   * l0
L_y = 100 * l0

# Setting Maximum Velocities of Phase Space Grid:
v_max_e = 0.0025  # Setting this value depending upon temperature. Can be later determined in terms of v0
v_max_i = 0.00027 # Setting this value depending upon temperature. Can be later determined in terms of v0

# Setting permeability:
c   = v_max_e # |c| units(c)
eps = 1 / (c**2 * mu)

# Velocity Scales:
thermal_speed   = velocity_scales.thermal_speed(T_background, m0, k0)
sound_speed     = velocity_scales.sound_speed(T_background, k0, gamma)
alfven_velocity = velocity_scales.alfven_velocity(B0, n_background, m0, mu) 

# Length scales:
debye_length = length_scales.debye_length(n_background, T_background, e0, k0, eps)
skin_depth   = length_scales.skin_depth(n_background, e0, c, m0, eps)
gyroradius   = length_scales.gyroradius(velocity_scales.thermal_speed(T_background, m0, k0), B0, e0, m0)

# Time scales:
plasma_frequency     = time_scales.plasma_frequency(n_background, e0, m0, eps)
cyclotron_frequency  = time_scales.cyclotron_frequency(B0, e0, m0)
alfven_crossing_time = time_scales.alfven_crossing_time(min(L_x, L_y), B0, n_background, m0, mu)
sound_crossing_time  = time_scales.sound_crossing_time(min(L_x, L_y), T_background, k0, gamma)

# Time parameters:
N_cfl   = 0.125
t_final = 500 * t0

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
PETSc.Sys.Print("Thermal Speed        :", thermal_speed)
PETSc.Sys.Print("Sound Speed          :", sound_speed)
PETSc.Sys.Print("Alfven Velocity      :", alfven_velocity)
PETSc.Sys.Print("Chosen Velocity Scale:", v0)
PETSc.Sys.Print("Maximum Velocity(e)  :", v_max_e / v0, "|v0| units(v0)")
PETSc.Sys.Print("Maximum Velocity(i)  :", v_max_i / v0, "|v0| units(v0)")
PETSc.Sys.Print("==================================================\n")

PETSc.Sys.Print("==================================================")
PETSc.Sys.Print("              Other Dependent Units               ")
PETSc.Sys.Print("==================================================")
PETSc.Sys.Print("Light Speed :", c / v0, "|v0| units(v0)")
PETSc.Sys.Print("Permittivity:", eps)
PETSc.Sys.Print("==================================================\n")

# Switch for solver components:
fields_enabled           = True
source_enabled           = True
hybrid_model_enabled     = False

# File-writing Parameters:
# Set to zero for no file-writing
dt_dump_f       = 1 * t0
# ALWAYS set dump moments and dump fields at same frequency:
dt_dump_moments = dt_dump_fields = 0.1 * t0

# Restart(Set to zero for no-restart):
t_restart = 0 * t0

# Variation of collisional-timescale parameter through phase space:
@af.broadcast
def tau(q1, q2, p1, p2, p3):
    return (tau_collisions * t0 * p1**0 * q1**0)
