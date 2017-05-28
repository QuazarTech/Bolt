"""
This module contains functions which are used to initialize
quantities, which will be used in the simulation. These quantities
are typically set using the parameters which are set in the config
object. As the name suggests, these functions only need to be 
called once during the simulation run.
"""

import numpy as np
import arrayfire as af 

def calculate_x(da, config):
  """
  Returns the 4D array of x which has the variations in x along axis 1
  
  Parameters:
  -----------
    da : This is an object of type PETSc.DMDA and is used in domain decomposition.
         The da object is used to refer to the local zone of computation

    config: Object config which is obtained by 
            setup_simulation.configuration_object() is passed to this filee

  Output:
  -------
    x : Array holding the values of x
  """

  N_x       = config.N_x
  N_vel_x   = config.N_vel_x
  N_vel_y   = config.N_vel_y
  N_ghost   = config.N_ghost

  # Getting the step-size in x:
  x_start  = config.x_start
  x_end    = config.x_end
  length_x = x_end - x_start
  dx       = (length_x)/(N_x - 1)

  # Obtaining the left corner coordinates for the local zone considered:
  ((j_bottom_left, i_bottom_left), (N_y_local, N_x_local)) = da.getCorners()

  # Constructing x using the above data:
  i = i_bottom_left + np.arange(-N_ghost, N_x_local + N_ghost, 1)
  x = x_start + i * dx
  x = af.Array.as_type(af.to_array(x), af.Dtype.f64)

  # Reordering and tiling such that variation in x is along axis 1:
  x = af.tile(af.reorder(x), N_y_local + 2*N_ghost, 1, N_vel_y, N_vel_x)

  af.eval(x)
  return(x)

def calculate_y(da, config):
  """
  Returns the 4D array of y which has the variations in y along axis 0
  
  Parameters:
  -----------
    da : This is an object of type PETSc.DMDA and is used in domain decomposition.
         The da object is used to refer to the local zone of computation

    config: Object config which is obtained by 
            setup_simulation.configuration_object() is passed to this filee

  Output:
  -------
    y : Array holding the values of y
  """

  N_y       = config.N_y
  N_vel_x   = config.N_vel_x
  N_vel_y   = config.N_vel_y
  N_ghost   = config.N_ghost

  # Getting the step-size in x:
  y_start  = config.y_start
  y_end    = config.y_end
  length_y = y_end - y_start
  dy       = (length_y)/(N_y - 1)

  # Obtaining the left corner coordinates for the local zone considered:
  ((j_bottom_left, i_bottom_left), (N_y_local, N_x_local)) = da.getCorners()

  # Constructing y using the above data:
  j = j_bottom_left + np.arange(-N_ghost, N_y_local + N_ghost, 1)
  y = y_start + j * dy
  y = af.Array.as_type(af.to_array(y), af.Dtype.f64)

  # y is tiled such that variation in y is along axis 0:
  y = af.tile(y, 1, N_x_local + 2*N_ghost, N_vel_y, N_vel_x)

  af.eval(y)
  return(y)

def calculate_vel_x(da, config):
  """
  Returns the 4D array of vel_x which has the variations in vel_x along axis 3
  
  Parameters:
  -----------
    da : This is an object of type PETSc.DMDA and is used in domain decomposition.
         The da object is used to refer to the local zone of computation

    config: Object config which is obtained by 
            setup_simulation.configuration_object() is passed to this filee

  Output:
  -------
    vel_x : Array holding the values of vel_x
  """

  N_vel_x = config.N_vel_x
  N_vel_y = config.N_vel_y
  N_ghost = config.N_ghost

  vel_x_max = config.vel_x_max

  # Obtaining the left corner coordinates for the local zone considered:
  ((j_bottom_left, i_bottom_left), (N_y_local, N_x_local)) = da.getCorners()

  # Constructing vel_x using the above data:
  i_v_x = np.arange(0, N_vel_x, 1)
  dv_x  = (2*vel_x_max)/(N_vel_x - 1)
  vel_x = -vel_x_max + i_v_x * dv_x
  vel_x = af.Array.as_type(af.to_array(vel_x), af.Dtype.f64)

  # Reordering and tiling such that variation in x-velocity is along axis 3
  vel_x = af.reorder(vel_x, 3, 2, 1, 0)
  vel_x = af.tile(vel_x,\
                  N_y_local + 2*N_ghost, \
                  N_x_local + 2*N_ghost, \
                  N_vel_y, \
                  1 
                 )

  af.eval(vel_x)
  return(vel_x)

def calculate_vel_y(da, config):
  """
  Returns the 4D array of vel_y which has the variations in vel_y along axis 2
  
  Parameters:
  -----------
    da : This is an object of type PETSc.DMDA and is used in domain decomposition.
         The da object is used to refer to the local zone of computation

    config: Object config which is obtained by 
            setup_simulation.configuration_object() is passed to this filee

  Output:
  -------
    vel_y : Array holding the values of vel_y
  """

  N_vel_x = config.N_vel_x
  N_vel_y = config.N_vel_y
  N_ghost = config.N_ghost

  vel_y_max = config.vel_y_max

  # Obtaining the left corner coordinates for the local zone considered:
  ((j_bottom_left, i_bottom_left), (N_y_local, N_x_local)) = da.getCorners()

  # Constructing vel_y using the above data:
  i_v_y = np.arange(0, N_vel_y, 1)
  dv_y  = (2*vel_y_max)/(N_vel_y - 1)
  vel_y = -vel_y_max + i_v_y * dv_y
  vel_y = af.Array.as_type(af.to_array(vel_y), af.Dtype.f64)

  # Reordering and tiling such that variation in y-velocity is along axis 2
  vel_y = af.reorder(vel_y, 3, 2, 0, 1)
  vel_y = af.tile(vel_y,\
                  N_y_local + 2*N_ghost, \
                  N_x_local + 2*N_ghost, \
                  1, \
                  N_vel_x 
                 )

  af.eval(vel_y)
  return(vel_y)

def f_background(da, config):
  """
  Returns the value of f_background, depending on the parameters set in 
  the config object
  
  Parameters:
  -----------
    da : This is an object of type PETSc.DMDA and is used in domain decomposition.
         The da object is used to refer to the local zone of computation

    config: Object config which is obtained by 
            setup_simulation.configuration_object() is passed to this filee

  Output:
  -------
    f_background : Array which contains the values of f_background at different 
                   values of (y, x, vel_y, vel_x)
  """
  
  mass_particle      = config.mass_particle
  boltzmann_constant = config.boltzmann_constant

  rho_background         = config.rho_background
  temperature_background = config.temperature_background
  vel_bulk_x_background  = config.vel_bulk_x_background
  vel_bulk_y_background  = config.vel_bulk_y_background

  # Calculating vel_x and vel_y for the local zone:
  vel_x = calculate_vel_x(da, config)
  vel_y = calculate_vel_y(da, config)

  # Depending on the dimensionality in velocity space, the 
  # distribution function is assigned accordingly:
  if(config.mode == '2V'):

    f_background = rho_background * \
                   (mass_particle/(2*np.pi*boltzmann_constant*temperature_background)) * \
                   af.exp(-mass_particle*(vel_x - vel_bulk_x_background)**2/\
                         (2*boltzmann_constant*temperature_background)) * \
                   af.exp(-mass_particle*(vel_y - vel_bulk_y_background)**2/\
                         (2*boltzmann_constant*temperature_background))

  else:
    f_background = rho_background *\
                   np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*temperature_background)) * \
                   af.exp(-mass_particle*(vel_x - vel_bulk_x_background)**2/\
                         (2*boltzmann_constant*temperature_background))
  
  af.eval(f_background)
  return(f_background)

def f_initial(da, config):
  """
  Returns the value of f_initial, depending on the parameters set in 
  the config object
  
  Parameters:
  -----------
    da : This is an object of type PETSc.DMDA and is used in domain decomposition.
         The da object is used to refer to the local zone of computation

    config: Object config which is obtained by 
            setup_simulation.configuration_object() is passed to this file

  Output:
  -------
    f_initial : Array which contains the values of f_initial at different values
                of (y, x, vel_y, vel_x)
  """

  mass_particle      = config.mass_particle
  boltzmann_constant = config.boltzmann_constant

  rho_background         = config.rho_background
  temperature_background = config.temperature_background
  vel_bulk_x_background  = config.vel_bulk_x_background
  vel_bulk_y_background  = config.vel_bulk_y_background

  N_vel_x = config.N_vel_x
  N_vel_y = config.N_vel_y

  vel_y_max = config.vel_y_max
  vel_x_max = config.vel_x_max

  dv_x = (2*vel_x_max)/(N_vel_x - 1)
  dv_y = (2*vel_y_max)/(N_vel_y - 1)

  pert_real = config.pert_real
  pert_imag = config.pert_imag
 
  k_x = config.k_x
  k_y = config.k_y

  # Calculating x, y,vel_x and vel_y for the local zone:
  x     = calculate_x(da, config)
  vel_x = calculate_vel_x(da, config)
  y     = calculate_y(da, config)
  vel_y = calculate_vel_y(da, config)

  # Calculating the perturbed density:
  rho   = rho_background + (pert_real * af.cos(k_x*x + k_y*y) -\
                            pert_imag * af.sin(k_x*x + k_y*y)
                           )

  # Depending on the dimensionality in velocity space, the 
  # distribution function is assigned accordingly:
  if(config.mode == '2V'):

    f_initial = rho * (mass_particle/(2*np.pi*boltzmann_constant*temperature_background)) * \
                af.exp(-mass_particle*(vel_x - vel_bulk_x_background)**2/\
                      (2*boltzmann_constant*temperature_background)) * \
                af.exp(-mass_particle*(vel_y - vel_bulk_y_background)**2/\
                      (2*boltzmann_constant*temperature_background))

  else:

    f_initial = rho *\
                np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*temperature_background)) * \
                af.exp(-mass_particle*(vel_x - vel_bulk_x_background)**2/\
                      (2*boltzmann_constant*temperature_background))
    
  normalization = af.sum(f_background(da, config)) * dv_x * dv_y/(x.shape[0] * x.shape[1])
  f_initial     = f_initial/normalization

  af.eval(f_initial)
  return(f_initial)