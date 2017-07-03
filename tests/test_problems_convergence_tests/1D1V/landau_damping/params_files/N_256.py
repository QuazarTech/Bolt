import numpy as np

num_devices = 1
mode        = '1V'

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

background_ions = dict(
                       rho         = 1.0, 
                       temperature = 1.0, 
                       vel_bulk_x  = 0,
                       vel_bulk_y  = 0,
                       vel_bulk_z  = 0,
                      )

perturbation = dict(
                    pert_real = 1e-2, 
                    pert_imag = 0,
                    k_x       = 2*np.pi,
                    k_y       = 0 #4*np.pi 
                   ) 

position_space = dict(N_x     = 256,
                      x_start = 0,
                      x_end   = 1.0,

                      N_y     = 3,
                      y_start = 0,
                      y_end   = 1.0,
 
                      N_ghost = 3
                     )

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

velocity_space = dict(N_vel_x   = 256,
                      vel_x_max = 10.0, 

                      N_vel_y   = 1, 
                      vel_y_max = 10.0,

                      N_vel_z   = 1, 
                      vel_z_max = 10.0
                     )

time = dict(
            final_time   = 0.0001/16,
            dt           = 0.0001/16 #*(32/position_space['N_x'])
           )

EM_fields = dict(
                 charge_electron = -10,
                 charge_ion      = 10,
                 solver          = 'electrostatic'
                )

collisions = dict(
                  collision_operator = 'BGK',
                  tau                = 0.0001 #np.inf
                 )