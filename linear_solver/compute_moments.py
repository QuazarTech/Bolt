"""
All the function under this sub-module return the various
moments of velocity of the perturbed distribution function
"""

import numpy as np

def delta_rho_hat(config, delta_f_hat):
  return(np.sum(delta_f_hat) * config.dv_x * config.dv_y * config.dv_z)

def delta_mom_bulk_x_hat(config, delta_f_hat):
  return(np.sum(delta_f_hat * config.vel_x) * config.dv_x * config.dv_y * config.dv_z)

def delta_vel_bulk_x_hat(config, delta_f_hat):
  return(delta_mom_bulk_x_hat(config, delta_f_hat)/config.rho_background)

def delta_mom_bulk_y_hat(config, delta_f_hat):
  return(np.sum(delta_f_hat * config.vel_y) * config.dv_x * config.dv_y * config.dv_z)

def delta_vel_bulk_y_hat(config, delta_f_hat):
  return(delta_mom_bulk_y_hat(config, delta_f_hat)/config.rho_background)

def delta_mom_bulk_z_hat(config, delta_f_hat):
  return(np.sum(delta_f_hat * config.vel_z) * config.dv_x * config.dv_y * config.dv_z)

def delta_vel_bulk_z_hat(config, delta_f_hat):
  return(delta_mom_bulk_z_hat(config, delta_f_hat)/config.rho_background)

def delta_T_hat(config, delta_f_hat):
  dv_x = config.dv_x
  dv_y = config.dv_y
  dv_z = config.dv_z
  
  if(config.mode == '3V'):
    delta_T_hat   = np.sum(delta_f_hat * ((config.vel_x**2 + config.vel_y**2 + config.vel_z**2)/3 -\
                                           config.temperature_background
                                          )) * dv_x * dv_y * dv_z/config.rho_background
  elif(config.mode == '2V'):
    delta_T_hat   = np.sum(delta_f_hat * ((config.vel_x**2 + config.vel_y**2)/2 -\
                                          config.temperature_background
                                          )) * dv_x * dv_y * dv_z/config.rho_background

  else:
    delta_T_hat   = np.sum(delta_f_hat * (config.vel_x**2 -config.temperature_background
                                         )
                          ) * dv_x * dv_y * dv_z/config.rho_background

  return(delta_T_hat)
