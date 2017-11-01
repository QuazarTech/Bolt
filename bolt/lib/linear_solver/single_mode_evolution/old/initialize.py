import numpy as np 


def df_dv_background(config):
  """
  Returns the value of the derivative of f_background w.r.t the vel_x, vel_y 
  and vel_z 

  Parameters:
  -----------
    config : Object config which is obtained by setup_simulation() is passed to this file
 
  Output:
  -------
    dfdv_x_background : Array which contains the values of dfdv_x_background at different values
                        of vel_x stacked along axis 1
    dfdv_y_background : Array which contains the values of dfdv_y_background at different values
                        of vel_y stacked along axis 0
    dfdv_z_background : Array which contains the values of dfdv_z_background at different values
                        of vel_z stacked along axis 2
  """
  
  vel_x, vel_y, vel_z = init_velocities(config)
  
  dv_x = config.dv_x
  dv_y = config.dv_y
  dv_z = config.dv_z

  # Varies along axis (0, 1, 2) as (vel_y, vel_x, vel_z)
  f_background_local = f_background(config)

  if(config.N_vel_z == 1 and config.N_vel_y == 1):
    dfdv_x_background = np.gradient(f_background_local[0, :, 0], dv_x)
    dfdv_y_background = np.zeros_like(f_background_local)
    dfdv_z_background = np.zeros_like(f_background_local)

  elif(config.N_vel_z == 1):
    dfdv_x_background = np.gradient(f_background_local[:, :, 0], dv_y, dv_x)[1]
    dfdv_y_background = np.gradient(f_background_local[:, :, 0], dv_y, dv_x)[0]
    dfdv_z_background = np.zeros_like(f_background_local)
  
  else:
    dfdv_x_background = np.gradient(f_background_local, dv_y, dv_x, dv_z)[1]
    dfdv_y_background = np.gradient(f_background_local, dv_y, dv_x, dv_z)[0]
    dfdv_z_background = np.gradient(f_background_local, dv_y, dv_x, dv_z)[2]

  dfdv_x_background = dfdv_x_background.reshape([config.N_vel_y, config.N_vel_x, config.N_vel_z])
  dfdv_y_background = dfdv_y_background.reshape([config.N_vel_y, config.N_vel_x, config.N_vel_z])
  dfdv_z_background = dfdv_z_background.reshape([config.N_vel_y, config.N_vel_x, config.N_vel_z])

  return(dfdv_x_background, dfdv_y_background, dfdv_z_background)

def init_delta_f_hat(config):
  """
  Returns the initial value of delta_f_hat which is setup depending on
  the perturbation parameters set in config. This is a perturbation
  created in density of the system.

  Parameters:
  -----------
    config : Object config which is obtained by setup_simulation() is passed to 
             this file
  
  Output:
  -------
    delta_f_hat_initial : Array which contains the values of initial mode perturbation 
                          in the distribution function. It varies with vel_y along axis 0,
                          vel_x along axis 1, and vel_z along axis 2
  """
  pert_real = config.pert_real 
  pert_imag = config.pert_imag 

  # Varies along axis (0, 1, 2) as (vel_y, vel_x, vel_z)
  delta_f_hat_initial = pert_real*f_background(config) +\
                        pert_imag*f_background(config)*1j 

  return delta_f_hat_initial