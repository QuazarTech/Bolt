import numpy as np
import arrayfire as af

# Can be defined as 'electrostatic', 'user-defined'.
# The initial conditions need to be specified under initialize
# Ensure that the initial conditions specified satisfy
# Maxwell's constraint equations
fields_initialize = 'user-defined'

# Can be defined as 'electrostatic' and 'fdtd'
fields_type   = 'electrostatic'
fields_solver = 'SNES'

# Can be defined as 'strang' and 'lie'
time_splitting = 'strang'

# Method in q-space
solver_method_in_q = 'FVM'
solver_method_in_p = 'FVM'

reconstruction_method_in_q = 'weno5'
reconstruction_method_in_p = 'weno5'

riemann_solver = 'upwind-flux'

# Restart(Set to zero for no-restart):
t_restart = 0

# File-writing Parameters:
# Set to zero for no file-writing
dt_dump_f       = 0.1
dt_dump_moments = 0.01 

# Time parameters:
N_cfl   = 0.45
t_final = 10

# Dimensionality considered in velocity space:
p_dim = 2

# Number of devices(GPUs/Accelerators) on each node:
num_devices = 1

# Constants:
mass_particle      = 0.910938356 # x 1e-30 kg
h_bar              = 1.0545718e-4 # x aJ ps
boltzmann_constant = 1
charge_electron    = -0.160217662 # x aC
speed_of_light     = 300. # x [um/ps]
fermi_velocity     = speed_of_light/300
epsilon0           = 8.854187817 # x [aC^2 / (aJ um) ]

# Spatial quantities (will be initialized to shape = [q1, q2] in initalize.py)
mu          = None # chemical potential used in the e-ph operator
T           = None # Electron temperature used in the e-ph operator
mu_ee       = None # chemical potential used in the e-e operator
T_ee        = None # Electron temperature used in the e-e operator
vel_drift_x = None
vel_drift_y = None
phi         = None # Electric potential in the plane of graphene sheet

# Momentum quantities (will be initialized to shape = [p1*p2*p3] in initialize.py)
E_band   = None
vel_band = None

collision_nonlinear_iters = 3
electrostatic_solver_step = 10000

# Variation of collisional-timescale parameter through phase space:
@af.broadcast
def tau_defect(q1, q2, p1, p2, p3):
    return(1e-3 * q1**0 * p1**0)

@af.broadcast
def tau_ee(q1, q2, p1, p2, p3):
    return(np.inf * q1**0 * p1**0)

def tau(q1, q2, p1, p2, p3):
    return(tau_defect(q1, q2, p1, p2, p3))

def band_energy(p_x, p_y):
    
    p = af.sqrt(p_x**2. + p_y**2.)
    
    E_upper = p*fermi_velocity

    af.eval(E_upper)
    return(E_upper)

def band_velocity(p_x, p_y):

    p     = af.sqrt(p_x**2. + p_y**2.)
    p_hat = [p_x / (p + 1e-20), p_y / (p + 1e-20)]

    v_f   = fermi_velocity

    upper_band_velocity =  [ v_f * p_hat[0],  v_f * p_hat[1]]

    af.eval(upper_band_velocity[0], upper_band_velocity[1])
    return(upper_band_velocity)

