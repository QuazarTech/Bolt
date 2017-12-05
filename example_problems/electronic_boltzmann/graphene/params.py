import numpy as np
import arrayfire as af

# Can be defined as 'electrostatic', 'user-defined'.
# The initial conditions need to be specified under initialize
# Ensure that the initial conditions specified satisfy
# Maxwell's constraint equations
fields_initialize = 'user-defined'

# Can be defined as 'electrostatic' and 'fdtd'
fields_solver = 'electrostatic'

# Can be defined as 'strang' and 'lie'
time_splitting = 'strang'

# Dimensionality considered in velocity space:
p_dim = 2

# Number of devices(GPUs/Accelerators) on each node:
num_devices = 1

# Constants:
mass_particle      = 0.910938356 # x 1e-30 kg
h_bar              = 1.0545718e-4 # x aJ ps
boltzmann_constant = 1
charge_electron    = -0.160217662 # x aC
speed_of_light     = 300 # x [um/ps]
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

collision_nonlinear_iters = 5

# Variation of collisional-timescale parameter through phase space:
def tau_defect(q1, q2, p1, p2, p3):
    return (af.constant(1., q1.shape[0], q2.shape[1], 
                        p1.shape[2], dtype = af.Dtype.f64
                       )
           )

def tau_ee(q1, q2, p1, p2, p3):
    return (af.constant(np.inf, q1.shape[0], q2.shape[1], 
                        p1.shape[2], dtype = af.Dtype.f64
                       )
           )

def band_energy(p_x, p_y):
    
    p = af.sqrt(p_x**2. + p_y**2.)

    return(p*fermi_velocity)

def band_velocity(p_x, p_y):

    p     = af.sqrt(p_x**2. + p_y**2.)
    p_hat = [p_x / (p + 1e-20), p_y / (p + 1e-20)]

    v_f   = fermi_velocity

    upper_band_velocity =  [ v_f * p_hat[0],  v_f * p_hat[1]]

    return(upper_band_velocity)

