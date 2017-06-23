import numpy as np
import lts.initialize as initialize
from lts.collision_operators import BGK_collision_operator

def ddelta_f_hat_dt(config, delta_f_hat,\
                    delta_E_x_hat, delta_E_y_hat, delta_E_z_hat,\
                    delta_B_x_hat, delta_B_y_hat, delta_B_z_hat
                   ):
  """
  Returns the value of the derivative of the mode perturbation of the distribution 
  function, and the field quantities with respect to time. This is used to evolve 
  the system with time.
  Parameters:
  -----------
    config : Object config which is obtained by setup_simulation() is passed to this file

    The following arrays fed to this function are the result of the last time-step's integration.
    At t=0 the initial mode perturbation of the system is passed to this function:

    delta_f_hat: Mode perturbation of the distribution function is passed to the function.

    delta_E_x_hat: Mode perturbation of the electric field Ex is passed to the function.

    delta_E_y_hat: Mode perturbation of the electric field Ey is passed to the function.

    delta_E_z_hat: Mode perturbation of the electric field Ez is passed to the function.

    delta_B_x_hat: Mode perturbation of the magnetic field Bx is passed to the function..

    delta_B_y_hat: Mode perturbation of the magnetic field By is passed to the function.

    delta_B_y_hat: Mode perturbation of the magnetic field Bz is passed to the function.

  Output:
  -------
    ddelta_f_hat_dt : Array which contains the values of the derivative of the mode 
                      perturbation of the distribution function with respect to time.

    ddelta_E_x_hat_dt : Array which contains the values of the derivative of the mode 
                        perturbation of the E_x with respect to time.

    ddelta_E_y_hat_dt : Array which contains the values of the derivative of the mode 
                        perturbation of the E_y with respect to time.

    ddelta_E_z_hat_dt : Array which contains the values of the derivative of the mode 
                        perturbation of the E_z with respect to time.

    ddelta_B_x_hat_dt : Array which contains the values of the derivative of the mode 
                        perturbation of the B_x with respect to time.

    ddelta_B_y_hat_dt : Array which contains the values of the derivative of the mode 
                        perturbation of the B_y with respect to time.

    ddelta_B_z_hat_dt : Array which contains the values of the derivative of the mode 
                        perturbation of the B_z with respect to time.

  """
  vel_x, vel_y, vel_z = initialize.init_velocities(config)

  dv_x = config.dv_x
  dv_y = config.dv_y
  dv_z = config.dv_z

  k_x = config.k_x   
  k_y = config.k_y

  mass_particle   = config.mass_particle
  charge_electron = config.charge_electron

  delta_mom_bulk_x = np.sum(vel_x * delta_f_hat) * dv_x * dv_y * dv_z
  delta_mom_bulk_y = np.sum(vel_y * delta_f_hat) * dv_x * dv_y * dv_z
  delta_mom_bulk_z = np.sum(vel_z * delta_f_hat) * dv_x * dv_y * dv_z

  dfdv_x_background, dfdv_y_background, dfdv_z_background =\
  initialize.dfdv_r_background(config)

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
  
  return(ddelta_f_hat_dt,\
         ddelta_E_x_hat_dt, ddelta_E_y_hat_dt, ddelta_E_z_hat_dt,\
         ddelta_B_x_hat_dt, ddelta_B_y_hat_dt, ddelta_B_z_hat_dt
        )

def RK6_step(config, delta_f_hat, dt):
  """
  Evolves the various mode perturbation arrays by a single time-step by
  making use of the RK-6 time-stepping scheme:
  
  Parameters:
  -----------   
    config : Object config which is obtained by setup_simulation() is passed to 
             this file
 
    delta_f_hat : Array containing the values of the delta_f_hat at the time (t0 - dt).
                  (where t0 is the value as given by the time-stepping loop) 
 
    dt : Time-step size.

  Output:
  -------
    delta_f_hat : Array containing the values of the delta_f_hat at the time (t0).
                  (where t0 is the value as given by the time-stepping loop). 

  """
  global delta_E_x_hat, delta_E_y_hat, delta_E_z_hat
  global delta_B_x_hat, delta_B_y_hat, delta_B_z_hat

  k1 = ddelta_f_hat_dt(config, delta_f_hat,\
                       delta_E_x_hat, delta_E_y_hat, delta_E_z_hat,\
                       delta_B_x_hat, delta_B_y_hat, delta_B_z_hat
                      )
  
  k2 = ddelta_f_hat_dt(config, delta_f_hat + 0.25*k1[0]*dt,\
                       delta_E_x_hat + 0.25*k1[1]*dt, delta_E_y_hat + 0.25*k1[2]*dt,\
                       delta_E_z_hat + 0.25*k1[3]*dt, delta_B_x_hat + 0.25*k1[4]*dt,\
                       delta_B_y_hat + 0.25*k1[5]*dt, delta_B_z_hat + 0.25*k1[6]*dt
                      )
  
  k3 = ddelta_f_hat_dt(config, delta_f_hat + (3/32)*(k1[0]+3*k2[0])*dt,\
                       delta_E_x_hat + (3/32)*(k1[1]+3*k2[1])*dt,\
                       delta_E_y_hat + (3/32)*(k1[2]+3*k2[2])*dt,\
                       delta_E_z_hat + (3/32)*(k1[3]+3*k2[3])*dt,\
                       delta_B_x_hat + (3/32)*(k1[4]+3*k2[4])*dt,\
                       delta_B_y_hat + (3/32)*(k1[5]+3*k2[5])*dt,\
                       delta_B_z_hat + (3/32)*(k1[6]+3*k2[6])*dt
                      )

  k4 = ddelta_f_hat_dt(config, delta_f_hat + (12/2197)*(161*k1[0]-600*k2[0]+608*k3[0])*dt,\
                       delta_E_x_hat + (12/2197)*(161*k1[1]-600*k2[1]+608*k3[1])*dt,\
                       delta_E_y_hat + (12/2197)*(161*k1[2]-600*k2[2]+608*k3[2])*dt,\
                       delta_E_z_hat + (12/2197)*(161*k1[3]-600*k2[3]+608*k3[3])*dt,\
                       delta_B_x_hat + (12/2197)*(161*k1[4]-600*k2[4]+608*k3[4])*dt,\
                       delta_B_y_hat + (12/2197)*(161*k1[5]-600*k2[5]+608*k3[5])*dt,\
                       delta_B_z_hat + (12/2197)*(161*k1[6]-600*k2[6]+608*k3[6])*dt
                      )
  
  k5 = ddelta_f_hat_dt(config, delta_f_hat + (1/4104)*(8341*k1[0]-32832*k2[0]+29440*k3[0]-845*k4[0])*dt,\
                       delta_E_x_hat + (1/4104)*(8341*k1[1]-32832*k2[1]+29440*k3[1]-845*k4[1])*dt,\
                       delta_E_y_hat + (1/4104)*(8341*k1[2]-32832*k2[2]+29440*k3[2]-845*k4[2])*dt,\
                       delta_E_z_hat + (1/4104)*(8341*k1[3]-32832*k2[3]+29440*k3[3]-845*k4[3])*dt,\
                       delta_B_x_hat + (1/4104)*(8341*k1[4]-32832*k2[4]+29440*k3[4]-845*k4[4])*dt,\
                       delta_B_y_hat + (1/4104)*(8341*k1[5]-32832*k2[5]+29440*k3[5]-845*k4[5])*dt,\
                       delta_B_z_hat + (1/4104)*(8341*k1[6]-32832*k2[6]+29440*k3[6]-845*k4[6])*dt
                      )
 
  k6 = ddelta_f_hat_dt(config, delta_f_hat + \
                       (-(8/27)*k1[0]+2*k2[0]-(3544/2565)*k3[0]+(1859/4104)*k4[0]-(11/40)*k5[0])*dt,\
                       delta_E_x_hat + \
                       (-(8/27)*k1[1]+2*k2[1]-(3544/2565)*k3[1]+(1859/4104)*k4[1]-(11/40)*k5[1])*dt,\
                       delta_E_y_hat + \
                       (-(8/27)*k1[2]+2*k2[2]-(3544/2565)*k3[2]+(1859/4104)*k4[2]-(11/40)*k5[2])*dt,\
                       delta_E_z_hat + \
                       (-(8/27)*k1[3]+2*k2[3]-(3544/2565)*k3[3]+(1859/4104)*k4[3]-(11/40)*k5[3])*dt,\
                       delta_B_x_hat + \
                       (-(8/27)*k1[4]+2*k2[4]-(3544/2565)*k3[4]+(1859/4104)*k4[4]-(11/40)*k5[4])*dt,\
                       delta_B_y_hat + \
                       (-(8/27)*k1[5]+2*k2[5]-(3544/2565)*k3[5]+(1859/4104)*k4[5]-(11/40)*k5[5])*dt,\
                       delta_B_z_hat + \
                       (-(8/27)*k1[6]+2*k2[6]-(3544/2565)*k3[6]+(1859/4104)*k4[6]-(11/40)*k5[6])*dt
                      )

  delta_f_hat   += 1/5*((16/27)*k1[0]+(6656/2565)*k3[0]+(28561/11286)*k4[0]-(9/10)*k5[0]+(2/11)*k6[0])*dt

  delta_E_x_hat += 1/5*((16/27)*k1[1]+(6656/2565)*k3[1]+(28561/11286)*k4[1]-(9/10)*k5[1]+(2/11)*k6[1])*dt
  delta_E_y_hat += 1/5*((16/27)*k1[2]+(6656/2565)*k3[2]+(28561/11286)*k4[2]-(9/10)*k5[2]+(2/11)*k6[2])*dt
  delta_E_z_hat += 1/5*((16/27)*k1[3]+(6656/2565)*k3[3]+(28561/11286)*k4[3]-(9/10)*k5[3]+(2/11)*k6[3])*dt

  delta_B_x_hat += 1/5*((16/27)*k1[4]+(6656/2565)*k3[4]+(28561/11286)*k4[4]-(9/10)*k5[4]+(2/11)*k6[4])*dt
  delta_B_y_hat += 1/5*((16/27)*k1[5]+(6656/2565)*k3[5]+(28561/11286)*k4[5]-(9/10)*k5[5]+(2/11)*k6[5])*dt
  delta_B_z_hat += 1/5*((16/27)*k1[6]+(6656/2565)*k3[6]+(28561/11286)*k4[6]-(9/10)*k5[6]+(2/11)*k6[6])*dt

  return(delta_f_hat)

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

  dv_x = config.dv_x
  dv_y = config.dv_y
  dv_z = config.dv_z

  k_x = config.k_x   
  k_y = config.k_y

  density_data = np.zeros(time_array.size)

  global delta_E_x_hat, delta_E_y_hat, delta_E_z_hat
  global delta_B_x_hat, delta_B_y_hat, delta_B_z_hat
  
  charge_electron = config.charge_electron

  # Intializing for the electrostatic Case:
  delta_rho_hat = np.sum(delta_f_hat_initial) * dv_x * dv_y * dv_z
  delta_phi_hat = charge_electron * delta_rho_hat/(k_x**2 + k_y**2)
  
  delta_E_x_hat = -delta_phi_hat * (1j * k_x)
  delta_E_y_hat = -delta_phi_hat * (1j * k_y)
  delta_E_z_hat = 0
  
  delta_B_x_hat = 0 
  delta_B_y_hat = 0 
  delta_B_z_hat = 0 

  density_data[0] = np.abs(delta_rho_hat)

  delta_f_hat = delta_f_hat_initial

  for time_index, t0 in enumerate(time_array[1:]):

    if(time_index%1==0):
      print("Computing for Time =", t0)
    
    dt = time_array[1] - time_array[0]

    delta_f_hat = RK6_step(config, delta_f_hat, dt)

    delta_rho_hat                = np.sum(delta_f_hat) * dv_x * dv_y * dv_z
    density_data[time_index + 1] = np.abs(delta_rho_hat)

  return(density_data, delta_f_hat)