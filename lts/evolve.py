import numpy as np
from scipy.integrate import odeint
import lts.initialize as initialize
from lts.collision_operators import BGK_collision_operator

def ddelta_f_hat_dt(delta_f_hat, config):
  """
  Returns the value of the derivative of the mode perturbation of the distribution 
  function with respect to time. This is used to evolve the system with time.

  Parameters:
  -----------
    config : Object config which is obtained by set() is passed to this file

    delta_f_hat: Mode perturbation of the distribution function that is passed to the function.
                 The array fed to this function is the result of the last time-step's integration.
                 At t=0 the initial mode perturbation of the system as declared in the configuration
                 file is passed to this function.

  Output:
  -------
    ddelta_f_hat_dt : Array which contains the values of the derivative of the Fourier mode 
                      perturbation of the distribution function with respect to time.
  """
  mass_particle      = config.mass_particle
  boltzmann_constant = config.boltzmann_constant

  rho_background         = config.rho_background
  temperature_background = config.temperature_background
  
  vel_x_max = config.vel_x_max
  N_vel_x   = config.N_vel_x
  vel_x     = np.linspace(-vel_x_max, vel_x_max, N_vel_x)
  dv_x      = vel_x[1] - vel_x[0]
  
  k_x = config.k_x   
  
  if(config.mode == '2D2V'):
    vel_y_max    = config.vel_y_max
    N_vel_y      = config.N_vel_y
    vel_y        = np.linspace(-vel_y_max, vel_y_max, N_vel_y) 
    dv_y         = vel_y[1] - vel_y[0]
    vel_x, vel_y = np.meshgrid(vel_x, vel_y)
    k_y          = config.k_y   
    
  fields_enabled  = config.fields_enabled
  charge_particle = config.charge_particle

  collisions_enabled = config.collisions_enabled 
  tau                = config.tau
  dt                 = config.dt

  if(config.mode == '2D2V'):
    delta_rho_hat    = np.sum(delta_f_hat) * dv_x *dv_y
    delta_vel_bulk_x = np.sum(vel_x * delta_f_hat) * dv_x *dv_y/rho_background
    delta_vel_bulk_y = np.sum(vel_y * delta_f_hat) * dv_x *dv_y/rho_background

  elif(config.mode == '1D1V'):
    delta_rho_hat = np.sum(delta_f_hat) * dv_x

  if(fields_enabled!="True"):
    fields_term = 0

  else:
    dfdv_x_background = initialize.dfdv_x_background(config)

    if(config.mode == '2D2V'):
      dfdv_y_background = initialize.dfdv_y_background(config)

      global delta_E_x_hat, delta_E_y_hat, delta_E_z_hat
      global delta_B_x_hat, delta_B_y_hat, delta_B_z_hat



      fields_term = (1 / mass_particle) * (charge_particle * delta_E_x_hat + \
                                           0 * delta_B_z_hat   * vel_y \
                                          ) * dfdv_x_background  + \
                    (1 / mass_particle) * (charge_particle * delta_E_y_hat - \
                                           0 * delta_B_z_hat   * vel_x
                                          ) * dfdv_y_background
    
    elif(config.mode == '1D1V'):
      delta_E_x_hat     = -charge_particle * (delta_rho_hat)/(1j * k_x)
      fields_term       = (1 / mass_particle) * (charge_particle * delta_E_x_hat) * dfdv_x_background
      
  if(collisions_enabled!="True"):
    C_f = 0

  else:
    C_f   = BGK_collision_operator(config, delta_f_hat)

  if(config.mode == '2D2V'):
    ddelta_f_hat_dt = -1j * (k_x * vel_x + k_y * vel_y) * delta_f_hat + fields_term + C_f
  
  elif(config.mode == '1D1V'):
    ddelta_f_hat_dt = -1j * (k_x * vel_x) * delta_f_hat + fields_term + C_f

  return ddelta_f_hat_dt

def EM_fields_mode1(delta_E_z_hat, delta_B_x_hat, delta_B_y_hat,\
                    delta_J_z_hat, k_x, k_y, dt
                   ):
  delta_E_z_hat += (delta_B_y_hat * 1j * k_x - delta_B_x_hat * 1j * k_y) * dt -\
                   delta_J_z_hat * dt
  delta_B_x_hat += -(delta_E_z_hat * 1j * k_y)*dt
  delta_B_y_hat += (delta_E_z_hat * 1j * k_x)*dt

  return(delta_E_z_hat, delta_B_x_hat, delta_B_y_hat)

def EM_fields_mode2(delta_B_z_hat, delta_E_x_hat, delta_E_y_hat,\
                    delta_J_x_hat, delta_J_y_hat, k_x, k_y, dt
                   ):
  delta_E_x_hat += (delta_B_z_hat * 1j * k_y)*dt - delta_J_x_hat * dt
  delta_E_y_hat += -(delta_B_z_hat * 1j * k_x)*dt - delta_J_y_hat * dt
  delta_B_z_hat += -(delta_E_y_hat * 1j * k_x - delta_E_x_hat * 1j * k_y) * dt

  return(delta_B_z_hat, delta_E_x_hat, delta_E_y_hat) 

def EM_fields_evolve(delta_E_x_hat, delta_E_y_hat, delta_E_z_hat,\
                     delta_B_x_hat, delta_B_y_hat, delta_B_z_hat,\
                     delta_J_x_hat, delta_J_y_hat, delta_J_z_hat,\
                     dt, k_x, k_y
                    ):
  delta_E_z_hat, delta_B_x_hat, delta_B_y_hat = EM_fields_mode1(delta_E_z_hat, delta_B_x_hat,\
                                                                delta_B_y_hat, delta_J_z_hat,\
                                                                k_x, k_y, dt
                                                               )

  delta_B_z_hat, delta_E_x_hat, delta_E_y_hat = EM_fields_mode2(delta_B_z_hat, delta_E_x_hat,\
                                                                delta_E_y_hat, delta_J_x_hat,\
                                                                delta_J_y_hat, k_x, k_y, dt
                                                               )

  return (delta_E_x_hat, delta_E_y_hat, delta_E_z_hat,\
          delta_B_x_hat, delta_B_y_hat, delta_B_z_hat)

def RK4_step(config, delta_f_hat_initial, dt):

  k1 = ddelta_f_hat_dt(delta_f_hat_initial, config)
  k2 = ddelta_f_hat_dt(delta_f_hat_initial + 0.5*k1*dt, config)
  k3 = ddelta_f_hat_dt(delta_f_hat_initial + 0.5*k2*dt, config)
  k4 = ddelta_f_hat_dt(delta_f_hat_initial + k3*dt, config)

  return(delta_f_hat_initial + ((k1+2*k2+2*k3+k4)/6)*dt)

def time_integration(config, delta_f_hat_initial, time_array):
  """
  Performs the time integration for the simulation. This is the main function that
  evolves the system in time. The parameters this function evolves for are dictated
  by the parameters as has been set in the config object. Final distribution function
  and the array that shows the evolution of rho_hat is returned by this function.

  Parameters:
  -----------   
    config : Object config which is obtained by set() is passed to this file

    delta_f_hat_initial : Array containing the initial values of the delta_f_hat. The value
                          for this function is typically obtained from the appropriately named 
                          function from the initialize submodule.

    time_array : Array which consists of all the time points at which we are evolving the system.
                 Data such as the mode amplitude of the density perturbation is also computed at 
                 the time points.

  Output:
  -------
    density_data : The value of the amplitude of the mode expansion of the density perturbation computed at
                   the various points in time as declared in time_array

    new_delta_f_hat : This value that is returned by the function is the distribution function that is obtained at
                      the final time-step. This is particularly useful in cases where comparisons need to be made 
                      between results of the Cheng-Knorr and the linear theory codes.
  
  """
  vel_x_max = config.vel_x_max
  N_vel_x   = config.N_vel_x
  N_x       = config.N_x
  x         = np.linspace(0, 1, N_x)
  k_x       = config.k_x 
  vel_x     = np.linspace(-vel_x_max, vel_x_max, N_vel_x)
  dv_x      = vel_x[1] - vel_x[0]

  charge_particle = config.charge_particle

  if(config.mode == '2D2V'):
    vel_y_max = config.vel_y_max
    N_vel_y   = config.N_vel_y
    N_y       = config.N_y
    y         = np.linspace(0, 1, N_y)
    k_y       = config.k_y 
    vel_y     = np.linspace(-vel_y_max, vel_y_max, N_vel_y)
    dv_y      = vel_y[1] - vel_y[0]  
    
    vel_x, vel_y = np.meshgrid(vel_x, vel_y)
    x, y         = np.meshgrid(x, y)

  density_data  = np.zeros(time_array.size)
  global delta_B_z_hat
  global delta_E_x_hat
  global delta_E_y_hat
  delta_rho_hat = np.sum(delta_f_hat_initial) * dv_x *dv_y
  delta_phi_hat = charge_particle * delta_rho_hat/(k_x**2 + k_y**2)
  delta_E_x_hat = delta_phi_hat * (1j * k_x)
  delta_E_y_hat = delta_phi_hat * (1j * k_y)
  delta_B_z_hat = 0

  old_delta_f_hat = delta_f_hat_initial

  for time_index, t0 in enumerate(time_array):
    t0 = time_array[time_index]
    if (time_index == time_array.size - 1):
        break
    t1 = time_array[time_index + 1]
    dt = t1 - t0

    if(time_index != 0):
      delta_f_hat_initial = old_delta_f_hat.copy()

    new_delta_f_hat = RK4_step(config, delta_f_hat_initial, dt)
    if(config.mode == '2D2V'):
      delta_rho_hat            = np.sum(new_delta_f_hat)*dv_x*dv_y
      density_data[time_index] = np.max(delta_rho_hat.real * np.cos(k_x*x + k_y*y) - \
                                        delta_rho_hat.imag * np.sin(k_x*x + k_y*y)
                                       )

    elif(config.mode == '1D1V'):
      delta_rho_hat            = np.sum(new_delta_f_hat)*dv_x
      density_data[time_index] = np.max(delta_rho_hat.real * np.cos(k_x*x) -\
                                        delta_rho_hat.imag * np.sin(k_x*x)
                                       )

    old_delta_f_hat          = new_delta_f_hat.copy()

  return(density_data, new_delta_f_hat)