import numpy as np
import arrayfire as af 

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

  config.N_x            = params.configuration_space['N_x']
  config.N_ghost_x      = params.configuration_space['N_ghost_x']
  config.left_boundary  = params.configuration_space['left_boundary']
  config.right_boundary = params.configuration_space['right_boundary']
  
  config.N_y          = params.configuration_space['N_y']
  config.N_ghost_y    = params.configuration_space['N_ghost_y']
  config.bot_boundary = params.configuration_space['bot_boundary']
  config.top_boundary = params.configuration_space['top_boundary']

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

def calculate_x(config):

  N_x       = config.N_x
  N_ghost_x = config.N_ghost_x
  N_vel_x   = config.N_vel_x

  N_y       = config.N_y
  N_ghost_y = config.N_ghost_y
  N_vel_y   = config.N_vel_y

  left_boundary  = config.left_boundary
  right_boundary = config.right_boundary
  length_x       = right_boundary - left_boundary

  dx = (length_x)/(N_x - 1)
  i  = np.arange(-N_ghost_x, N_x + N_ghost_x, 1)
  x  = left_boundary + i * dx
  x  = af.Array.as_type(af.to_array(x), af.Dtype.f64)
  x  = af.tile(af.reorder(x), N_y + 2*N_ghost_y, 1, N_vel_y, N_vel_x)

  af.eval(x)
  return(x)

def calculate_y(config):

  N_x       = config.N_x
  N_ghost_x = config.N_ghost_x
  N_vel_x   = config.N_vel_x

  N_y       = config.N_y
  N_ghost_y = config.N_ghost_y
  N_vel_y   = config.N_vel_y

  bot_boundary = config.bot_boundary
  top_boundary = config.top_boundary
  length_y     = top_boundary - bot_boundary
  
  dy = (length_y)/(N_y - 1)
  j  = np.arange(-N_ghost_y, N_y + N_ghost_y, 1)
  y  = bot_boundary  + j * dy
  y  = af.Array.as_type(af.to_array(y), af.Dtype.f64)
  y  = af.tile(y, 1, N_x + 2*N_ghost_x, N_vel_y, N_vel_x)
  
  af.eval(y)
  return(y)

def calculate_vel_x(config):

  N_x       = config.N_x
  N_ghost_x = config.N_ghost_x
  N_vel_x   = config.N_vel_x

  N_y       = config.N_y
  N_ghost_y = config.N_ghost_y
  N_vel_y   = config.N_vel_y

  vel_x_max = config.vel_x_max

  i_v_x = np.arange(0, N_vel_x, 1)
  dv_x  = (2*vel_x_max)/(N_vel_x - 1)
  vel_x = -vel_x_max + i_v_x * dv_x
  vel_x = af.Array.as_type(af.to_array(vel_x), af.Dtype.f64)
  vel_x = af.reorder(vel_x, 3, 2, 1, 0)
  vel_x = af.tile(vel_x,\
                  N_y + 2*N_ghost_y, \
                  N_x + 2*N_ghost_x, \
                  N_vel_y, \
                  1 
                 )


  af.eval(vel_x)
  return(vel_x)

def calculate_vel_y(config):

  N_x       = config.N_x
  N_ghost_x = config.N_ghost_x
  N_vel_x   = config.N_vel_x

  N_y       = config.N_y
  N_ghost_y = config.N_ghost_y
  N_vel_y   = config.N_vel_y

  vel_y_max = config.vel_y_max

  i_v_y = np.arange(0, N_vel_y, 1)
  dv_y  = (2*vel_y_max)/(N_vel_y - 1)
  vel_y = -vel_y_max + i_v_y * dv_y
  vel_y = af.Array.as_type(af.to_array(vel_y), af.Dtype.f64)
  vel_y = af.reorder(vel_y, 3, 2, 0, 1)
  vel_y = af.tile(vel_y,\
                  N_y + 2*N_ghost_y, \
                  N_x + 2*N_ghost_x, \
                  1, \
                  N_vel_x 
                 )

  af.eval(vel_y)
  return(vel_y)


def f_background(config):
  
  mass_particle      = config.mass_particle
  boltzmann_constant = config.boltzmann_constant

  rho_background         = config.rho_background
  temperature_background = config.temperature_background
  vel_bulk_x_background  = config.vel_bulk_x_background
  vel_bulk_y_background  = config.vel_bulk_y_background

  vel_x = calculate_vel_x(config)
  vel_y = calculate_vel_y(config)

  if(config.mode == '2V'):

    f_background = rho_background *\
                   (mass_particle/(2*np.pi*\
                                   boltzmann_constant*\
                                   temperature_background
                                   )
                   ) * \
                   af.exp(-mass_particle*
                           (vel_x - vel_bulk_x_background)**2/\
                           (2*boltzmann_constant*temperature_background
                           )
                         ) * \
                   af.exp(-mass_particle*\
                           (vel_y - vel_bulk_y_background)**2/\
                           (2*boltzmann_constant*temperature_background
                           )
                         )

  else:
    f_background = rho_background * \
                   np.sqrt(mass_particle/\
                           (2*np.pi*boltzmann_constant*temperature_background
                           ) 
                          ) * \
                   af.exp(-mass_particle*(vel_x - vel_bulk_x_background)**2/\
                          (2*boltzmann_constant*temperature_background)
                         )
  
  af.eval(f_background)
  return(f_background)

def f_initial(config):

  mass_particle      = config.mass_particle
  boltzmann_constant = config.boltzmann_constant

  rho_background         = config.rho_background
  temperature_background = config.temperature_background
  vel_bulk_x_background  = config.vel_bulk_x_background
  vel_bulk_y_background  = config.vel_bulk_y_background

  pert_real = config.pert_real
  pert_imag = config.pert_imag
 
  k_x   = config.k_x
  k_y   = config.k_y

  x     = calculate_x(config)
  vel_x = calculate_vel_x(config)
  y     = calculate_y(config)
  vel_y = calculate_vel_y(config)

  if(config.mode == '2V'):
  
    rho   = rho_background + (pert_real * af.cos(k_x*x + k_y*y) -\
                              pert_imag * af.sin(k_x*x + k_y*y)
                             )

    f_initial = rho * (mass_particle/(2*np.pi*boltzmann_constant*temperature_background)) * \
                af.exp(-mass_particle*(vel_x - vel_bulk_x_background)**2/\
                        (2*boltzmann_constant*temperature_background)) * \
                af.exp(-mass_particle*(vel_y - vel_bulk_y_background)**2/\
                        (2*boltzmann_constant*temperature_background))

  else:
    rho = rho_background + (pert_real * af.cos(2*np.pi*x) -\
                            pert_imag * af.sin(2*np.pi*x)
                           )

    f_initial = rho *\
                np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*temperature_background)) * \
                af.exp(-mass_particle*(vel_x - vel_bulk_x_background)**2/\
                (2*boltzmann_constant*temperature_background))
  
  af.eval(f_initial)
  return(f_initial)

def time_array(config):

  final_time = config.final_time
  dt         = config.dt

  time_array = np.arange(dt, final_time + dt, dt)

  return(time_array)