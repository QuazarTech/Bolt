"""
This module contains functions which are used to initialize
quantities, which will be used in the simulation. These quantities
are typically set using the parameters which are set in the config
object. As the name suggests, these functions only need to be 
called once during the simulation run.
"""

import numpy as np
import arrayfire as af 

class options:
  def __init__(self):
    pass

def set(params):
  """
  Used to set the parameters that are used in the simulation

  Parameters:
  -----------
    params : Name of the file that contains the parameters for the 
             simulation run is passed to this function. 

  Output:
  -------
    config : Object whose attributes contain all the simulation parameters. 
             This is passed to the remaining solver functions.
  """
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
  """
  Returns the 2D/4D array of x depending on the dimensionlity of the 
  system considered, which is used in the computations of the Cheng-Knorr code.

  Parameters:
  -----------
    config : Object config which is obtained by set() is passed to this file

  Output:
  -------
    x : Array holding the values of x
  """
  N_x       = config.N_x
  N_ghost_x = config.N_ghost_x
  N_vel_x   = config.N_vel_x

  if(config.mode == '2D2V'):
    N_y       = config.N_y
    N_ghost_y = config.N_ghost_y
    N_vel_y   = config.N_vel_y

  left_boundary  = config.left_boundary
  right_boundary = config.right_boundary

  x  = np.linspace(left_boundary, right_boundary, N_x)
  dx = x[1] - x[0]

  x_ghost_left  = np.linspace(-(N_ghost_x)*dx + left_boundary,\
                               left_boundary - dx,\
                               N_ghost_x
                             )

  x_ghost_right = np.linspace(right_boundary + dx,\
                              right_boundary + N_ghost_x*dx ,\
                              N_ghost_x
                             )

  x = np.concatenate([x_ghost_left, x, x_ghost_right])
  x = af.Array.as_type(af.to_array(x), af.Dtype.f64)

  if(config.mode == '2D2V'):
    x = af.tile(af.reorder(x), N_y + 2*N_ghost_y, 1, N_vel_x, N_vel_y)

  else:
    x  = af.tile(x, 1, N_vel_x)

  af.eval(x)
  return(x)

def calculate_y(config):
  """
  Returns the 4D array of y which is used in the computations of the 
  Cheng-Knorr code.

  Parameters:
  -----------
    config : Object config which is obtained by set() is passed to this 
             file

  Output:
  -------
    y : Array holding the values of y
  """
  if(config.mode != '2D2V'):
    raise Exception('Not in 2D mode!')

  N_x       = config.N_x
  N_ghost_x = config.N_ghost_x
  N_vel_x   = config.N_vel_x
  N_y       = config.N_y
  N_ghost_y = config.N_ghost_y
  N_vel_y   = config.N_vel_y

  bot_boundary = config.bot_boundary
  top_boundary = config.top_boundary

  y  = np.linspace(bot_boundary, top_boundary, N_y)
  dy = y[1] - y[0]

  y_ghost_bot = np.linspace(-(N_ghost_y)*dy + bot_boundary,\
                              bot_boundary - dy,\
                              N_ghost_y
                           )

  y_ghost_top = np.linspace(top_boundary + dy,\
                            top_boundary + N_ghost_y*dy,\
                            N_ghost_y
                           )

  y = np.concatenate([y_ghost_bot, y, y_ghost_top])
  y = af.Array.as_type(af.to_array(y), af.Dtype.f64)
  y = af.tile(y, 1, N_x + 2*N_ghost_x, N_vel_x, N_vel_y)
  
  af.eval(y)
  return(y)

def calculate_vel_x(config):
  """
  Returns the 2D/4D array of vel_x depending on the dimensionality 
  of the system considered, which is used in the computations of the 
  Cheng-Knorr code.

  Parameters:
  -----------
    config : Object config which is obtained by set() is passed to 
             this file

  Output:
  -------
    vel_x : Array holding the values of vel_x
  """
  N_x       = config.N_x
  N_ghost_x = config.N_ghost_x
  N_vel_x   = config.N_vel_x

  if(config.mode == '2D2V'):
    N_y       = config.N_y
    N_ghost_y = config.N_ghost_y
    N_vel_y   = config.N_vel_y

  vel_x_max = config.vel_x_max

  vel_x = np.linspace(-vel_x_max, vel_x_max, N_vel_x)
  vel_x = af.Array.as_type(af.to_array(vel_x), af.Dtype.f64)

  if(config.mode == '2D2V'):
    vel_x = af.reorder(vel_x, 3, 2, 0, 1)
    vel_x = af.tile(vel_x, N_y + 2*N_ghost_y,\
                    N_x + 2*N_ghost_x, 1, N_vel_y
                   )

  else:
    vel_x = af.tile(af.reorder(vel_x), N_x + 2*N_ghost_x, 1)

  af.eval(vel_x)
  return(vel_x)

def calculate_vel_y(config):
  """
  Returns the 4D array of vel_y which is used in the computations of 
  the Cheng-Knorr code.

  Parameters:
  -----------
    config : Object config which is obtained by set() is passed to this 
             file

  Output:
  -------
    vel_y : Array holding the values of vel_y
  """
  if(config.mode != '2D2V'):
    raise Exception('Not in 2D mode!')

  N_x       = config.N_x
  N_ghost_x = config.N_ghost_x
  N_vel_x   = config.N_vel_x
  N_y       = config.N_y
  N_ghost_y = config.N_ghost_y
  N_vel_y   = config.N_vel_y

  vel_y_max = config.vel_y_max

  vel_y = np.linspace(-vel_y_max, vel_y_max, N_vel_y)
  vel_y = af.Array.as_type(af.to_array(vel_y), af.Dtype.f64)
  vel_y = af.reorder(vel_y, 3, 2, 1, 0)
  vel_y = af.tile(vel_y, N_y + 2*N_ghost_y,\
                  N_x + 2*N_ghost_x, N_vel_x, 1
                 )

  af.eval(vel_y)
  return(vel_y)


def f_background(config):
  """
  Returns the value of f_background, depending on the parameters set in 
  the config object

  Parameters:
  -----------
    config : Object config which is obtained by set() is passed to this file

  Output:
  -------
    f_background : Array which contains the values of f_background at different 
                   values of (x, y, vel_x, vel_y) in the 4D case. In the 2D case, 
                   f_background is calculated at different (x, vel_x)
  """
  
  mass_particle      = config.mass_particle
  boltzmann_constant = config.boltzmann_constant

  rho_background         = config.rho_background
  temperature_background = config.temperature_background
  vel_bulk_x_background  = config.vel_bulk_x_background

  vel_x = calculate_vel_x(config)

  if(config.mode == '2D2V'):
    vel_y = calculate_vel_y(config)
    vel_bulk_y_background  = config.vel_bulk_y_background

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
  """
  Returns the value of f_initial, depending on the parameters set in 
  the config object

  Parameters:
  -----------
    config : Object config which is obtained by set() is passed to this file

  Output:
  -------
    f_initial : Array which contains the values of f_initial at different values
                of (x, y, vel_x, vel_y) in the 4D case. In the 2D case, f_initial 
                is calculated at different (x, vel_x)
  """
  
  mass_particle      = config.mass_particle
  boltzmann_constant = config.boltzmann_constant

  rho_background         = config.rho_background
  temperature_background = config.temperature_background
  
  vel_bulk_x_background  = config.vel_bulk_x_background

  pert_real = config.pert_real
  pert_imag = config.pert_imag
 
  k_x   = config.k_x
  x     = calculate_x(config)
  vel_x = calculate_vel_x(config)

  if(config.mode == '2D2V'):
    k_y                    = config.k_y
    vel_bulk_y_background  = config.vel_bulk_y_background
    y                      = calculate_y(config)
    vel_y                  = calculate_vel_y(config)
    
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
  """
  Returns the value of the time_array at which we solve for in the 
  simulation. The time_array is set depending on the options which 
  have been mention in config.

  Parameters:
  -----------
    config : Object config which is obtained by set() is passed to 
             this file

  Output:
  -------
    time_array : Array that contains the values of time at which 
                 the simulation evaluates the physical quantities. 
  """

  final_time = config.final_time
  dt         = config.dt

  time_array = np.arange(dt, final_time + dt, dt)

  return(time_array)