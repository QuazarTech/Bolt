import numpy as np 

def f_background(config):

  mass_particle      = config.mass_particle
  boltzmann_constant = config.boltzmann_constant

  rho_background         = config.rho_background
  temperature_background = config.temperature_background
  vel_bulk_x_background  = config.vel_bulk_x_background
  vel_bulk_y_background  = config.vel_bulk_y_background

  vel_x_max = config.vel_x_max
  N_vel_x   = config.N_vel_x
  vel_x     = np.linspace(-vel_x_max, vel_x_max, N_vel_x)

  vel_y_max = config.vel_y_max
  N_vel_y   = config.N_vel_y
  vel_y     = np.linspace(-vel_y_max, vel_y_max, N_vel_y)

  dv_x  = vel_x[1] - vel_x[0]
  dv_y  = vel_y[1] - vel_y[0]

  vel_x, vel_y = np.meshgrid(vel_x, vel_y)

  if(config.mode == '2V'):    
    f_background = rho_background * \
                   (mass_particle/(2*np.pi*boltzmann_constant*temperature_background)) * \
                   np.exp(-mass_particle*(vel_x - vel_bulk_x_background)**2/\
                         (2*boltzmann_constant*temperature_background)) * \
                   np.exp(-mass_particle*(vel_y - vel_bulk_y_background)**2/\
                         (2*boltzmann_constant*temperature_background))

  elif(config.mode == '1V'):
    f_background = rho_background * \
                   np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*temperature_background)) * \
                   np.exp(-mass_particle*(vel_x - vel_bulk_x_background)**2/\
                         (2*boltzmann_constant*temperature_background))

  # normalization = np.sum(f_background) * dv_x * dv_y
  # print(normalization)
  # f_background  = f_background #/normalization
  return f_background

# def dfdv_x_background(config):

#   vel_x_max = config.vel_x_max
#   N_vel_x   = config.N_vel_x
#   dv_x      = (2*vel_x_max)/(N_vel_x - 1)

#   f_background_local = f_background(config)
#   dfdv_x_background  = np.zeros([f_background_local.shape[0], f_background_local.shape[1]])

#   for i in range(f_background_local.shape[0]):
#     dfdv_x_background[i] = np.convolve(f_background_local[i], [1, -1], 'same') * (1/dv_x)

#   return dfdv_x_background

# def dfdv_y_background(config):

#   vel_y_max = config.vel_y_max
#   N_vel_y   = config.N_vel_y
#   dv_y      = (2*vel_y_max)/(N_vel_y - 1)

#   f_background_local = f_background(config)
#   dfdv_y_background  = np.zeros([f_background_local.shape[0], f_background_local.shape[1]])

#   for i in range(f_background_local.shape[1]):
#     dfdv_y_background[:, i] = np.convolve(f_background_local[:, i], [1, -1], 'same') * (1/dv_y)

#   return dfdv_y_background

def init_delta_f_hat(config):

  pert_real = config.pert_real 
  pert_imag = config.pert_imag 

  delta_f_hat_initial = pert_real*f_background(config) +\
                        pert_imag*f_background(config)*1j 

  return delta_f_hat_initial