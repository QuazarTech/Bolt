import numpy as np 

def maxwell_boltzmann(config, vel_x, vel_y, vel_z):
  mass_particle      = config.mass_particle
  boltzmann_constant = config.boltzmann_constant

  rho_background         = config.rho_background
  temperature_background = config.temperature_background
  vel_bulk_x_background  = config.vel_bulk_x_background
  vel_bulk_y_background  = config.vel_bulk_y_background
  vel_bulk_z_background  = config.vel_bulk_z_background

  if(config.mode == '3V'):    
    f = rho_background * \
        (mass_particle/(2*np.pi*boltzmann_constant*temperature_background))**(3/2) * \
        np.exp(-mass_particle*(vel_x - vel_bulk_x_background)**2/\
              (2*boltzmann_constant*temperature_background)) * \
        np.exp(-mass_particle*(vel_y - vel_bulk_y_background)**2/\
              (2*boltzmann_constant*temperature_background)) * \
        np.exp(-mass_particle*(vel_z - vel_bulk_z_background)**2/\
              (2*boltzmann_constant*temperature_background))


  elif(config.mode == '2V'):    
    f = rho_background * \
        (mass_particle/(2*np.pi*boltzmann_constant*temperature_background)) * \
        np.exp(-mass_particle*(vel_x - vel_bulk_x_background)**2/\
              (2*boltzmann_constant*temperature_background)) * \
        np.exp(-mass_particle*(vel_y - vel_bulk_y_background)**2/\
              (2*boltzmann_constant*temperature_background))

  elif(config.mode == '1V'):
    f = rho_background * \
        np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*temperature_background)) * \
        np.exp(-mass_particle*(vel_x - vel_bulk_x_background)**2/\
              (2*boltzmann_constant*temperature_background))

  return(f)

def init_velocities(config):
  # These are the cell centered values in velocity space:
  i_v_x = 0.5 + np.arange(0, config.N_vel_x, 1)
  dv_x  = (2*config.vel_x_max)/config.N_vel_x
  vel_x = -config.vel_x_max + i_v_x * dv_x

  i_v_y = 0.5 + np.arange(0, config.N_vel_y, 1)
  dv_y  = (2*config.vel_y_max)/config.N_vel_y
  vel_y = -config.vel_y_max + i_v_y * dv_y

  i_v_z = 0.5 + np.arange(0, config.N_vel_z, 1)
  dv_z  = (2*config.vel_z_max)/config.N_vel_z
  vel_z = -config.vel_z_max + i_v_z * dv_z

  vel_x, vel_y, vel_z = np.meshgrid(vel_x, vel_y, vel_z)

  vel_x = vel_x.reshape([config.N_vel_y, config.N_vel_x, config.N_vel_z])
  vel_y = vel_y.reshape([config.N_vel_y, config.N_vel_x, config.N_vel_z])
  vel_z = vel_z.reshape([config.N_vel_y, config.N_vel_x, config.N_vel_z])
 
  return(vel_x, vel_y, vel_z)

def f_background(config, return_normalization = 0):

  vel_x, vel_y, vel_z = init_velocities(config)

  dv_x = (2*config.vel_x_max)/config.N_vel_x
  dv_y = (2*config.vel_y_max)/config.N_vel_y
  dv_z = (2*config.vel_z_max)/config.N_vel_z
  
  f_background  = maxwell_boltzmann(config, vel_x, vel_y, vel_z)
  normalization = np.sum(f_background) * dv_x * dv_y * dv_z
  f_background  = f_background/normalization

  if(return_normalization == 1):
    return normalization
  
  else:
    return f_background

def dfdv_r_background(config):
  
  vel_x, vel_y, vel_z = init_velocities(config)
  
  dv_x = (2*config.vel_x_max)/config.N_vel_x
  dv_y = (2*config.vel_y_max)/config.N_vel_y
  dv_z = (2*config.vel_z_max)/config.N_vel_z

  f_background_local = f_background(config)

  if(config.N_vel_z == 1 and config.N_vel_y == 1):
    dfdv_x_background = (np.gradient(f_background_local[0, :, 0]))/dv_x
    dfdv_y_background = np.zeros_like(f_background_local)
    dfdv_z_background = np.zeros_like(f_background_local)

  elif(config.N_vel_z == 1):
    dfdv_x_background = (np.gradient(f_background_local[:, :, 0])[1])/dv_x
    dfdv_y_background = (np.gradient(f_background_local[:, :, 0])[0])/dv_y
    dfdv_z_background = np.zeros_like(f_background_local)
  
  else:
    dfdv_x_background = (np.gradient(f_background_local)[1])/dv_x
    dfdv_y_background = (np.gradient(f_background_local)[0])/dv_y
    dfdv_z_background = (np.gradient(f_background_local)[2])/dv_z

  dfdv_x_background = dfdv_x_background.reshape([config.N_vel_y, config.N_vel_x, config.N_vel_z])
  dfdv_y_background = dfdv_y_background.reshape([config.N_vel_y, config.N_vel_x, config.N_vel_z])
  dfdv_z_background = dfdv_z_background.reshape([config.N_vel_y, config.N_vel_x, config.N_vel_z])

  return(dfdv_x_background, dfdv_y_background, dfdv_z_background)

def init_delta_f_hat(config):

  pert_real = config.pert_real 
  pert_imag = config.pert_imag 

  delta_f_hat_initial = pert_real*f_background(config) +\
                        pert_imag*f_background(config)*1j 

  return delta_f_hat_initial