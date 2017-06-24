import numpy as np
import lts.initialize as initialize
import lts.compute_moments
import lts.timestepper
from lts.collision_operators import BGK_collision_operator

def dY_dt(config, Y0):
  """
  Returns the value of the derivative of the mode perturbation of the distribution 
  function, and the field quantities with respect to time. This is used to evolve 
  the system with time.

  Parameters:
  -----------
    config : Object config which is obtained by setup_simulation() is passed to this file

    The following arrays fed to this function are the result of the last time-step's integration.
    At t=0 the initial mode perturbation of the system is passed to this function:

  Output:
  -------

  """
  vel_x, vel_y, vel_z = initialize.init_velocities(config)

  k_x = config.k_x   
  k_y = config.k_y

  mass_particle   = config.mass_particle
  charge_electron = config.charge_electron

  delta_f_hat   = Y0[0]
  delta_E_x_hat = Y0[1]
  delta_E_y_hat = Y0[2]
  delta_E_z_hat = Y0[3]
  delta_B_x_hat = Y0[4]
  delta_B_y_hat = Y0[5]
  delta_B_z_hat = Y0[6]

  delta_mom_bulk_x = lts.compute_moments.delta_mom_bulk_x_hat(config, delta_f_hat)
  delta_mom_bulk_y = lts.compute_moments.delta_mom_bulk_y_hat(config, delta_f_hat)
  delta_mom_bulk_z = lts.compute_moments.delta_mom_bulk_z_hat(config, delta_f_hat)

  dfdv_x_background, dfdv_y_background, dfdv_z_background =\
    config.dfdv_x_background, config.dfdv_y_background, config.dfdv_z_background

  delta_J_x_hat = charge_electron * delta_mom_bulk_x
  delta_J_y_hat = charge_electron * delta_mom_bulk_y
  delta_J_z_hat = charge_electron * delta_mom_bulk_z
  
  ddelta_E_x_hat_dt = 0 #(delta_B_z_hat * 1j * k_y) - delta_J_x_hat
  ddelta_E_y_hat_dt = 0 #(- delta_B_z_hat * 1j * k_x) - delta_J_y_hat
  ddelta_E_z_hat_dt = 0 #(delta_B_y_hat * 1j * k_x - delta_B_x_hat * 1j * k_y) - delta_J_z_hat

  ddelta_B_x_hat_dt = 0 #(- delta_E_z_hat * 1j * k_y)
  ddelta_B_y_hat_dt = 0 #(delta_E_z_hat * 1j * k_x)
  ddelta_B_z_hat_dt = 0 #(delta_E_x_hat * 1j * k_y - delta_E_y_hat * 1j * k_x)

  fields_term = (charge_electron / mass_particle) * (delta_E_x_hat + \
                                                     delta_B_z_hat * vel_y - \
                                                     delta_B_y_hat * vel_z
                                                    ) * dfdv_x_background  + \
                (charge_electron / mass_particle) * (delta_E_y_hat + \
                                                     delta_B_x_hat * vel_z - \
                                                     delta_B_z_hat * vel_x
                                                    ) * dfdv_y_background  + \
                (charge_electron / mass_particle) * (delta_E_z_hat + \
                                                     delta_B_y_hat * vel_x -\
                                                     delta_B_x_hat * vel_y
                                                    ) * dfdv_z_background

  C_f = BGK_collision_operator(config, delta_f_hat)

  ddelta_f_hat_dt = -1j * (k_x * vel_x + k_y * vel_y) * delta_f_hat -\
                     fields_term + C_f 
  
  dY_dt = np.array([ddelta_f_hat_dt,\
                    ddelta_E_x_hat_dt, ddelta_E_y_hat_dt, ddelta_E_z_hat_dt,\
                    ddelta_B_x_hat_dt, ddelta_B_y_hat_dt, ddelta_B_z_hat_dt])
  
  return(dY_dt)

def compute_electrostatic_fields(config, delta_f_hat):

  # Intializing for the electrostatic Case:
  delta_rho_hat = lts.compute_moments.delta_rho_hat(config, delta_f_hat)
  delta_phi_hat = config.charge_electron * delta_rho_hat/(config.k_x**2 + config.k_y**2)
  
  delta_E_x_hat = -delta_phi_hat * (1j * config.k_x)
  delta_E_y_hat = -delta_phi_hat * (1j * config.k_y)
  delta_E_z_hat = 0
  
  delta_B_x_hat = 0 
  delta_B_y_hat = 0 
  delta_B_z_hat = 0 

  return(delta_E_x_hat, delta_E_y_hat, delta_E_z_hat,\
         delta_B_x_hat, delta_B_y_hat, delta_B_z_hat
        )

def time_integration(config, delta_f_hat_initial, time_array):
  """
  Performs the time integration for the simulation. This is the main function that
  evolves the system in time. The parameters this function evolves for are dictated
  by the parameters as has been set in the config object. Final distribution function
  and the array that shows the evolution of abs(rho_hat) is returned by this function.
 
  Parameters:
  -----------   
    config : Object config which is obtained by setup_simulation() is passed to this file
 
    delta_f_hat_initial : Array containing the initial values of the delta_f_hat. The value
                          for this function is typically obtained from the appropriately named 
                          function from the initialize submodule.
 
    time_array : Array which consists of all the time points at which we are evolving the system.
                 Data such as the mode amplitude of the density perturbation is also computed at 
                 the time points.
  Output:
  -------
    density_data : The value of the amplitude of the mode expansion of the density perturbation 
                   computed at the various points in time as declared in time_array
 
    new_delta_f_hat : delta_f_hat array at the final time-step. This is then passed to the export 
                      function to perform comparisons with the Cheng-Knorr code.

  """
  vel_x, vel_y, vel_z = initialize.init_velocities(config)

  # Adding the arrays of vel_x, vel_y and vel_z to config for faster access:
  # This is to avoid creation of the velocity arrays multiple times:
  config.vel_x, config.vel_y, config.vel_z = vel_x, vel_y, vel_z
  
  # Similarly, we add the arrays of dfdv_x, dfdv_y and dfdv_z:
  config.dfdv_x_background, config.dfdv_y_background, config.dfdv_z_background = \
    lts.initialize.df_dv_background(config)

  density_data    = np.zeros(time_array.size)
  density_data[0] = np.abs(lts.compute_moments.delta_rho_hat(config, delta_f_hat_initial))

  delta_f_hat   = delta_f_hat_initial
  delta_E_x_hat = compute_electrostatic_fields(config, delta_f_hat)[0]
  delta_E_y_hat = compute_electrostatic_fields(config, delta_f_hat)[1]
  delta_E_z_hat = compute_electrostatic_fields(config, delta_f_hat)[2]
  delta_B_x_hat = compute_electrostatic_fields(config, delta_f_hat)[3]
  delta_B_y_hat = compute_electrostatic_fields(config, delta_f_hat)[4]
  delta_B_z_hat = compute_electrostatic_fields(config, delta_f_hat)[5]

  Y = np.array([delta_f_hat, \
                delta_E_x_hat, delta_E_y_hat, delta_E_z_hat, \
                delta_B_x_hat, delta_B_y_hat, delta_B_z_hat])

  for time_index, t0 in enumerate(time_array[1:]):

    if(time_index%1==0):
      print("Computing for Time =", t0)
    
    dt = time_array[1] - time_array[0]

    Y                            = lts.timestepper.RK6_step(config, dY_dt, Y, dt)
    density_data[time_index + 1] = np.abs(lts.compute_moments.delta_rho_hat(config, Y[0]))

  return(density_data, delta_f_hat)