import numpy as np
import lts.initialize as initialize
from lts.collision_operators import BGK_collision_operator

def ddelta_f_hat_dt(config, delta_f_hat,\
                    delta_E_x_hat, delta_E_y_hat, delta_E_z_hat,\
                    delta_B_x_hat, delta_B_y_hat, delta_B_z_hat
                   ):

  vel_x, vel_y, vel_z = initialize.init_velocities(config)

  dv_x = vel_x[0, 1, 0] - vel_x[0, 0, 0]
  dv_y = vel_y[1, 0, 0] - vel_y[0, 0, 0]
  dv_z = vel_z[0, 0, 1] - vel_z[0, 0, 0]

  k_x = config.k_x   
  k_y = config.k_y
  k_z = config.k_z

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
  
  ddelta_E_x_hat_dt = (delta_B_z_hat * 1j * k_y - delta_B_y_hat * 1j * k_z) - delta_J_x_hat
  ddelta_E_y_hat_dt = (delta_B_x_hat * 1j * k_z - delta_B_z_hat * 1j * k_x) - delta_J_y_hat
  ddelta_E_z_hat_dt = (delta_B_y_hat * 1j * k_x - delta_B_x_hat * 1j * k_y) - delta_J_z_hat

  ddelta_B_x_hat_dt = (delta_E_y_hat * 1j * k_z - delta_E_z_hat * 1j * k_y)
  ddelta_B_y_hat_dt = (delta_E_z_hat * 1j * k_x - delta_E_x_hat * 1j * k_z)
  ddelta_B_z_hat_dt = (delta_E_x_hat * 1j * k_y - delta_E_y_hat * 1j * k_x)

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

  ddelta_f_hat_dt = -1j * (k_x * vel_x + k_y * vel_y + k_z * vel_z) * delta_f_hat -\
                     fields_term + C_f
  
  return(ddelta_f_hat_dt,\
         ddelta_E_x_hat_dt, ddelta_E_y_hat_dt, ddelta_E_z_hat_dt,\
         ddelta_B_x_hat_dt, ddelta_B_y_hat_dt, ddelta_B_z_hat_dt
        )

def RK4_step(config, delta_f_hat, dt):
  
  global delta_E_x_hat, delta_E_y_hat, delta_E_z_hat
  global delta_B_x_hat, delta_B_y_hat, delta_B_z_hat

  k1 = ddelta_f_hat_dt(config, delta_f_hat,\
                       delta_E_x_hat, delta_E_y_hat, delta_E_z_hat,\
                       delta_B_x_hat, delta_B_y_hat, delta_B_z_hat
                      )
  
  k2 = ddelta_f_hat_dt(config, delta_f_hat + 0.5*k1[0]*dt,\
                       delta_E_x_hat + 0.5*k1[1]*dt, delta_E_y_hat + 0.5*k1[2]*dt,\
                       delta_E_z_hat + 0.5*k1[3]*dt, delta_B_x_hat + 0.5*k1[4]*dt,\
                       delta_B_y_hat + 0.5*k1[5]*dt, delta_B_z_hat + 0.5*k1[6]*dt
                      )
  
  k3 = ddelta_f_hat_dt(config, delta_f_hat + 0.5*k2[0]*dt,\
                       delta_E_x_hat + 0.5*k2[1]*dt, delta_E_y_hat + 0.5*k2[2]*dt,\
                       delta_E_z_hat + 0.5*k2[3]*dt, delta_B_x_hat + 0.5*k2[4]*dt,\
                       delta_B_y_hat + 0.5*k2[5]*dt, delta_B_z_hat + 0.5*k2[6]*dt
                      )

  k4 = ddelta_f_hat_dt(config, delta_f_hat + 0.5*k3[0]*dt,\
                       delta_E_x_hat + 0.5*k3[1]*dt, delta_E_y_hat + 0.5*k3[2]*dt,\
                       delta_E_z_hat + 0.5*k3[3]*dt, delta_B_x_hat + 0.5*k3[4]*dt,\
                       delta_B_y_hat + 0.5*k3[5]*dt, delta_B_z_hat + 0.5*k3[6]*dt
                      )

  delta_f_hat   += ((k1[0]+2*k2[0]+2*k3[0]+k4[0])/6)*dt

  delta_E_x_hat += ((k1[1]+2*k2[1]+2*k3[1]+k4[1])/6)*dt
  delta_E_y_hat += ((k1[2]+2*k2[2]+2*k3[2]+k4[2])/6)*dt
  delta_E_z_hat += ((k1[3]+2*k2[3]+2*k3[3]+k4[3])/6)*dt

  delta_B_x_hat += ((k1[4]+2*k2[4]+2*k3[4]+k4[5])/6)*dt
  delta_B_y_hat += ((k1[5]+2*k2[5]+2*k3[5]+k4[4])/6)*dt
  delta_B_z_hat += ((k1[6]+2*k2[6]+2*k3[6]+k4[6])/6)*dt

  return(delta_f_hat)

def time_integration(config, delta_f_hat_initial, time_array):

  vel_x, vel_y, vel_z = initialize.init_velocities(config)

  dv_x = vel_x[0, 1, 0] - vel_x[0, 0, 0]
  dv_y = vel_y[1, 0, 0] - vel_y[0, 0, 0]
  dv_z = vel_z[0, 0, 1] - vel_z[0, 0, 0]

  k_x = config.k_x   
  k_y = config.k_y
  k_z = config.k_z 

  density_data  = np.zeros(time_array.size)

  global delta_E_x_hat, delta_E_y_hat, delta_E_z_hat
  global delta_B_x_hat, delta_B_y_hat, delta_B_z_hat
  
  charge_electron = config.charge_electron

  # Electrostatic Case:
  delta_rho_hat = np.sum(delta_f_hat_initial) * dv_x * dv_y * dv_z
  delta_phi_hat = charge_electron * delta_rho_hat/(k_x**2 + k_y**2 + k_z**2)
  
  delta_E_x_hat = -delta_phi_hat * (1j * k_x)
  delta_E_y_hat = -delta_phi_hat * (1j * k_y)
  delta_E_z_hat = -delta_phi_hat * (1j * k_z)
  
  delta_B_x_hat = 0 
  delta_B_y_hat = 0 
  delta_B_z_hat = 0 

  density_data[0] = np.abs(delta_rho_hat)

  delta_f_hat = delta_f_hat_initial

  for time_index, t0 in enumerate(time_array[1:]):

    if(time_index%1==0):
      print("Computing for Time =", t0)
    
    dt = time_array[1] - time_array[0]

    delta_f_hat = RK4_step(config, delta_f_hat, dt)

    delta_rho_hat                = np.sum(delta_f_hat) * dv_x * dv_y * dv_z
    density_data[time_index + 1] = np.abs(delta_rho_hat)

  return(density_data, delta_f_hat)