import numpy as np 

class options:
  def __init__(self):
    pass

def set(params):

  config = options()

  config.mode = params.mode

  config.mass_particle      = params.constants['mass_particle']
  config.boltzmann_constant = params.constants['boltzmann_constant']

  config.rho_background         = params.background_electrons['rho']
  config.temperature_background = params.background_electrons['temperature']
  config.vel_bulk_x_background  = params.background_electrons['vel_bulk_x']
  config.vel_bulk_y_background  = params.background_electrons['vel_bulk_y']

  config.pert_real = params.perturbation['pert_real']
  config.pert_imag = params.perturbation['pert_imag']
  config.k_x       = params.perturbation['k_x']
  config.k_y       = params.perturbation['k_y']

  config.N_x       = params.configuration_space['N_x']
  config.x_start   = params.configuration_space['x_start']
  config.x_end     = params.configuration_space['x_end']
  
  config.N_y       = params.configuration_space['N_y']
  config.y_start   = params.configuration_space['y_start']
  config.y_end     = params.configuration_space['y_end']

  config.N_ghost = params.configuration_space['N_ghost']

  config.N_vel_x   = params.velocity_space['N_vel_x']
  config.vel_x_max = params.velocity_space['vel_x_max']
  config.N_vel_y   = params.velocity_space['N_vel_y']
  config.vel_y_max = params.velocity_space['vel_y_max']

  config.bc_in_x = params.boundary_conditions['in_x']
  config.bc_in_y = params.boundary_conditions['in_y']

  config.left_rho         = params.boundary_conditions['left_rho']  
  config.left_temperature = params.boundary_conditions['left_temperature']
  config.left_vel_bulk_x  = params.boundary_conditions['left_vel_bulk_x']
  config.left_vel_bulk_y  = params.boundary_conditions['left_vel_bulk_y']
  
  config.right_rho         = params.boundary_conditions['right_rho']  
  config.right_temperature = params.boundary_conditions['right_temperature']
  config.right_vel_bulk_x  = params.boundary_conditions['right_vel_bulk_x']
  config.right_vel_bulk_y  = params.boundary_conditions['right_vel_bulk_y']

  config.bot_rho         = params.boundary_conditions['bot_rho']  
  config.bot_temperature = params.boundary_conditions['bot_temperature']
  config.bot_vel_bulk_x  = params.boundary_conditions['bot_vel_bulk_x']
  config.bot_vel_bulk_y  = params.boundary_conditions['bot_vel_bulk_y']
  
  config.top_rho         = params.boundary_conditions['top_rho']  
  config.top_temperature = params.boundary_conditions['top_temperature']
  config.top_vel_bulk_x  = params.boundary_conditions['top_vel_bulk_x']
  config.top_vel_bulk_y  = params.boundary_conditions['top_vel_bulk_y']

  config.final_time = params.time['final_time']
  config.dt         = params.time['dt']
    
  config.charge_particle = params.EM_fields['charge_particle']

  config.collision_operator = params.collisions['collision_operator']
  config.tau                = params.collisions['tau']

  return config


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

  vel_x, vel_y = np.meshgrid(vel_x, vel_y)

  if(config.mode == '2V'):    
    f_background = rho_background * (mass_particle/(2*np.pi*boltzmann_constant*temperature_background)) * \
                   np.exp(-mass_particle*(vel_x - vel_bulk_x_background)**2/(2*boltzmann_constant*temperature_background)) * \
                   np.exp(-mass_particle*(vel_y - vel_bulk_y_background)**2/(2*boltzmann_constant*temperature_background))

  elif(config.mode == '1V'):
    f_background = rho_background * np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*temperature_background)) * \
                   np.exp(-mass_particle*vel_x**2/(2*boltzmann_constant*temperature_background))
  

  return f_background

def dfdv_x_background(config):

  vel_x_max = config.vel_x_max
  N_vel_x   = config.N_vel_x

  vel_x = np.linspace(-vel_x_max, vel_x_max, N_vel_x)
  dv_x  = vel_x[1] - vel_x[0]

  vel_y_max = config.vel_y_max
  N_vel_y   = config.N_vel_y

  vel_y = np.linspace(-vel_y_max, vel_y_max, N_vel_y)
  dv_y  = vel_y[1] - vel_y[0]

  normalization      = np.sum(f_background(config)) * dv_x * dv_y
  f_background_local = f_background(config)/normalization
  dfdv_x_background  = np.zeros([f_background_local.shape[0], f_background_local.shape[1]])

  for i in range(f_background_local.shape[0]):
    dfdv_x_background[i] = np.convolve(f_background_local[i], [1, -1], 'same') * (1/dv_x)

  return dfdv_x_background

def dfdv_y_background(config):

  vel_x_max = config.vel_x_max
  N_vel_x   = config.N_vel_x

  vel_x = np.linspace(-vel_x_max, vel_x_max, N_vel_x)
  dv_x  = vel_x[1] - vel_x[0]

  vel_y_max = config.vel_y_max
  N_vel_y   = config.N_vel_y

  vel_y = np.linspace(-vel_y_max, vel_y_max, N_vel_y)
  dv_y  = vel_y[1] - vel_y[0]

  normalization      = np.sum(f_background(config)) * dv_x * dv_y
  f_background_local = f_background(config)/normalization
  dfdv_y_background  = np.zeros([f_background_local.shape[0], f_background_local.shape[1]])

  for i in range(f_background_local.shape[1]):
    dfdv_y_background[:, i] = np.convolve(f_background_local[:, i], [1, -1], 'same') * (1/dv_y)

  return dfdv_y_background

def time_array(config):

  final_time = config.final_time
  dt         = config.dt

  time_array = np.arange(0, final_time + dt, dt)

  return time_array

def init_delta_f_hat(config):

  pert_real = config.pert_real 
  pert_imag = config.pert_imag 

  delta_f_hat_initial = pert_real*f_background(config) +\
                        pert_imag*f_background(config)*1j 
  
  vel_x_max = config.vel_x_max
  N_vel_x   = config.N_vel_x
  dv_x      = (2*vel_x_max)/(N_vel_x - 1)

  vel_y_max = config.vel_y_max
  N_vel_y   = config.N_vel_y
  dv_y      = (2*vel_y_max)/(N_vel_y - 1)
 
  normalization       = np.sum(f_background(config)) * dv_x * dv_y
  delta_f_hat_initial = delta_f_hat_initial/normalization

  return delta_f_hat_initial