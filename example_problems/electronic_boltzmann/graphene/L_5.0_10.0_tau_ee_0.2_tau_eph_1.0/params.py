import numpy as np
import arrayfire as af

instantaneous_collisions = False #TODO : Remove from lib
hybrid_model_enabled = False #TODO : Remove from lib
source_enabled = True

fields_enabled = False
# Can be defined as 'electrostatic', 'user-defined'.
# The initial conditions need to be specified under initialize
# Ensure that the initial conditions specified satisfy
# Maxwell's constraint equations
fields_initialize = 'user-defined'

# Can be defined as 'electrostatic' and 'fdtd'
# To turn feedback from Electric fields on, set fields_solver = 'LCA'
# and set charge_electron
fields_type   = 'electrostatic'
fields_solver = 'SNES'

# Can be defined as 'strang' and 'lie'
time_splitting = 'strang'

# Method in q-space
solver_method_in_q = 'FVM'
solver_method_in_p = 'FVM'

reconstruction_method_in_q = 'minmod'
reconstruction_method_in_p = 'minmod'

riemann_solver_in_q = 'upwind-flux'
riemann_solver_in_p = 'upwind-flux'

# Restart(Set to zero for no-restart):
restart = 0
restart_file = '/home/mani/work/quazar_research/bolt/example_problems/electronic_boltzmann/graphene/dumps/f_eqbm.h5'
phi_restart_file = '/home/mani/work/quazar_research/bolt/example_problems/electronic_boltzmann/graphene/dumps/phi_eqbm.h5'
electrostatic_solver_every_nth_step = 1000000
solve_for_equilibrium = 0


# File-writing Parameters:
dump_steps = 10

# Time parameters:
dt      = 0.025/2 # ps
t_final = 200.     # ps

# Dimensionality considered in velocity space:
p_dim = 2
p_space_grid = 'cartesian' # Supports 'cartesian' or 'polar' grids

# Number of devices(GPUs/Accelerators) on each node:
num_devices = 6

# Constants:
mass_particle      = 0.910938356 # x 1e-30 kg
h_bar              = 1.0545718e-4 # x aJ ps
boltzmann_constant = 1
charge             = [0.*-0.160217662] # x aC
mass               = [0.] #TODO : Not used in electronic_boltzmann
                          # Remove from lib
speed_of_light     = 300. # x [um/ps]
fermi_velocity     = speed_of_light/300
epsilon0           = 8.854187817 # x [aC^2 / (aJ um) ]

epsilon_relative      = 3.9 # SiO2
backgate_potential    = -10 # V
global_chem_potential = 0.03
contact_start         = 4.5 # um
contact_end           = 5.5 # um
contact_geometry      = "straight" # Contacts on either side of the device
                                   # For contacts on the same side, use 
                                   # contact_geometry = "turn_around"

initial_temperature = 12e-4
initial_mu          = 0.015
vel_drift_x_in      = 1e-4*fermi_velocity
vel_drift_x_out     = 1e-4*fermi_velocity
AC_freq             = 1./100 # ps^-1

B3_mean = 1. # T

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

collision_operator_nonlinear_iters  = 2

# Variation of collisional-timescale parameter through phase space:
@af.broadcast
def tau_defect(q1, q2, p1, p2, p3):
    return(1.0 * q1**0 * p1**0)

@af.broadcast
def tau_ee(q1, q2, p1, p2, p3):
    return(0.2 * q1**0 * p1**0)

def tau(q1, q2, p1, p2, p3):
    return(tau_defect(q1, q2, p1, p2, p3))

def band_energy(p1, p2):

    if (p_space_grid == 'cartesian'):
        p_x = p1
        p_y = p2
    else : 
        raise NotImplementedError('Unsupported coordinate system in p_space')
    
    p = af.sqrt(p_x**2. + p_y**2.)
    
    E_upper = p*fermi_velocity

    af.eval(E_upper)
    return(E_upper)

def band_velocity(p1, p2):

    if (p_space_grid == 'cartesian'):
        p_x = p1
        p_y = p2
    else : 
        raise NotImplementedError('Unsupported coordinate system in p_space') 
    
    p     = af.sqrt(p_x**2. + p_y**2.)
    p_hat = [p_x / (p + 1e-20), p_y / (p + 1e-20)]

    v_f   = fermi_velocity

    upper_band_velocity =  [ v_f * p_hat[0],  v_f * p_hat[1]]

    af.eval(upper_band_velocity[0], upper_band_velocity[1])
    return(upper_band_velocity)

@af.broadcast
def fermi_dirac(mu, E_band):

    k = boltzmann_constant
    T = initial_temperature

    f = (1./(af.exp( (E_band - mu
                     )/(k*T) 
                   ) + 1.
            )
        )

    af.eval(f)
    return(f)
