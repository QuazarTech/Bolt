import numpy as np 

def maxwell_boltzmann(config, vel_x, vel_y, vel_z):
  """
  Returns the MB distribution for the parameters that have been set in
  the config object.
  
  Parameters:
  -----------
    config : Object config which is obtained by setup_simulation() is passed to this file

    vel_x : 3D velocity array which contains the variation in x-velocity stacked 
            along axis 1
    vel_y : 3D velocity array which contains the variation in y-velocity stacked 
            along axis 0
    vel_z : 3D velocity array which contains the variation in z-velocity stacked 
            along axis 2
  
  Output:
  -------
    f : The Maxwell-Boltzmann distribution function declared with the parameters that have been
        set for the background quantities in config.

  """

  m = config.mass_particle
  k = config.boltzmann_constant

  rho      = config.rho_background
  T        = config.temperature_background
  v_x_bulk = config.vel_bulk_x_background
  v_y_bulk = config.vel_bulk_y_background
  v_z_bulk = config.vel_bulk_z_background

  if(config.mode == '3V'):    
    f = rho * (m/(2*np.pi*k*T))**(3/2) * \
          np.exp(-m*(vel_x - v_x_bulk)**2/(2*k*T)) * \
            np.exp(-m*(vel_y - v_y_bulk)**2/(2*k*T)) * \
              np.exp(-m*(vel_z - v_z_bulk)**2/(2*k*T))


  elif(config.mode == '2V'):    
    f = rho * (m/(2*np.pi*k*T)) * \
          np.exp(-m*(vel_x - v_x_bulk)**2/(2*k*T)) * \
            np.exp(-m*(vel_y - v_y_bulk)**2/(2*k*T))

  elif(config.mode == '1V'):
    f = rho * np.sqrt(m/(2*np.pi*k*T)) * \
          np.exp(-m*(vel_x - v_x_bulk)**2/(2*k*T))

  return(f)

def init_velocities(config):
  """
  This function is used to initialized the velocity arrays vel_x, vel_y and 
  vel_z.

  Parameters:
  -----------
    config : Object config which is obtained by setup_simulation() is passed to this file

  Output:
  -------
    vel_x : 3D velocity array which contains the variation in x-velocity stacked 
            along axis 1
    vel_y : 3D velocity array which contains the variation in y-velocity stacked 
            along axis 0
    vel_z : 3D velocity array which contains the variation in z-velocity stacked 
            along axis 2

  """
  # These are the cell centered values in velocity space:
  i_v_x = 0.5 + np.arange(0, config.N_vel_x, 1)
  vel_x = -config.vel_x_max + i_v_x * config.dv_x

  i_v_y = 0.5 + np.arange(0, config.N_vel_y, 1)
  vel_y = -config.vel_y_max + i_v_y * config.dv_y

  i_v_z = 0.5 + np.arange(0, config.N_vel_z, 1)
  vel_z = -config.vel_z_max + i_v_z * config.dv_z

  vel_x, vel_y, vel_z = np.meshgrid(vel_x, vel_y, vel_z)

  vel_x = vel_x.reshape([config.N_vel_y, config.N_vel_x, config.N_vel_z])
  vel_y = vel_y.reshape([config.N_vel_y, config.N_vel_x, config.N_vel_z])
  vel_z = vel_z.reshape([config.N_vel_y, config.N_vel_x, config.N_vel_z])
 
  return(vel_x, vel_y, vel_z)

def f_background(config, return_normalization = 0):
  """
  Returns the value of f_background, depending on the parameters set in 
  the config object
  
  Parameters:
  -----------
    config : Object config which is obtained by set() is passed to this file
  
  Output:
  -------
    f_background : Array which contains the values of f_background at different values
                   of vel_x.
  """
  vel_x, vel_y, vel_z = init_velocities(config)

  f_background  = maxwell_boltzmann(config, vel_x, vel_y, vel_z)
  normalization = np.sum(f_background) * config.dv_x * config.dv_y * config.dv_z
  f_background  = f_background/normalization

  if(return_normalization == 1):
    return normalization
  
  else:
    return f_background

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
  the perturbation parameters set in config. 

  Parameters:
  -----------
    config : Object config which is obtained by setup_simulation() is passed to 
             this file
  
  Output:
  -------
    delta_f_hat_initial : Array which contains the values of initial mode perturbation 
                          in the distribution function.
  """
  pert_real = config.pert_real 
  pert_imag = config.pert_imag 

  delta_f_hat_initial = pert_real*f_background(config) +\
                        pert_imag*f_background(config)*1j 

  return delta_f_hat_initial