import numpy as np

# Change num_device if the node you are running on contains more than one device
# For instance, when running on a node which contains more than one GPU/Xeon-Phi

num_devices = 1

# Mode is used to indicate the dimensionality which has been considered in velocity space.
mode = '1V'

constants = dict(
                  mass_particle      = 1.0,
                  boltzmann_constant = 1.0,
                )

background_electrons = dict(
                            rho         = 1.0, 
                            temperature = 1.0, 
                            vel_bulk_x  = 0,
                            vel_bulk_y  = 0,
                            vel_bulk_z  = 0,
                           )

# NOT-IMPLEMENTED(In Development)
background_ions = dict(
                       rho         = 1.0, 
                       temperature = 1.0, 
                       vel_bulk_x  = 0,
                       vel_bulk_y  = 0,
                       vel_bulk_z  = 0,
                      )

# These are perturbations created in density
# k_x and k_y are the wave numbers of the sinusoidal perturbations
# in the x and y directions respectively.
perturbation = dict(
                    pert_real = 0.01, 
                    pert_imag = 0,
                    k_x       = 2*np.pi,
                    k_y       = 0*np.pi,\
                   ) 

# Resolution in position space:
position_space = dict(N_x     = 32,
                      x_start = 0,
                      x_end   = 1.0,

                      N_y     = 3,
                      y_start = 0,
                      y_end   = 1.0,

                      N_ghost = 3
                     )

# Boundary conditions can be changed to 'dirichlet' as well
# However fields haven't been implemented with Dirichlet B.C's
boundary_conditions = dict(in_x = 'periodic',
                           in_y = 'periodic',

                           # Mention for Dirichlet Below:
                           left_temperature = 1.0,
                           left_rho         = 1.0,
                           left_vel_bulk_x  = 0,
                           left_vel_bulk_y  = 0,

                           right_temperature = 1.0,
                           right_rho         = 1.0,
                           right_vel_bulk_x  = 0,
                           right_vel_bulk_y  = 0, 

                           bot_temperature = 1.0,
                           bot_rho         = 1.0,
                           bot_vel_bulk_x  = 0,
                           bot_vel_bulk_y  = 0, 

                           top_temperature = 1.0,
                           top_rho         = 1.0,
                           top_vel_bulk_x  = 0,
                           top_vel_bulk_y  = 0
                          )

# Resolution in velocity space:
velocity_space = dict(N_vel_x   = 64,
                      vel_x_max = 9.0, 

                      N_vel_y   = 1, 
                      vel_y_max = 9.0,

                      N_vel_z   = 1, 
                      vel_z_max = 9.0
                     )

time = dict(
            final_time   = 1.0,
            dt           = 0.01
           )

# charge_ion makes no difference currently(In development)
EM_fields = dict(
                 charge_electron = -10,
                 charge_ion      = 10, 
                 solver          = 'electrostatic'
                )

# Only BGK collision operator has been implemented so far.
collisions = dict(
                  collision_operator = 'BGK',
                  tau                = 0.01 #np.inf
                 )