import numpy as np
import arrayfire as af 

def calculate_x_left(da, config):
  N_x     = config.N_x
  N_ghost = config.N_ghost

  N_vel_x = config.N_vel_x
  N_vel_y = config.N_vel_y

  # Getting the step-size in x:
  x_start  = config.x_start
  x_end    = config.x_end
  length_x = x_end - x_start
  dx       = length_x/N_x
  
  # Obtaining the left-bottom corner coordinates 
  # of the left-bottom corner cell in the local zone considered:
  ((j_bottom, i_left), (N_y_local, N_x_local)) = da.getCorners()

  # Constructing x_left using the above data:
  i      = i_left  + np.arange(-N_ghost, N_x_local + N_ghost, 1)
  x_left = x_start + i * dx
  x_left = af.Array.as_type(af.to_array(x_left), af.Dtype.f64)

  # Reordering and tiling such that variation in x is along axis 1:
  x_left = af.tile(af.reorder(x_left), N_y_local + 2*N_ghost, 1, N_vel_y, N_vel_x)

  af.eval(x_left)
  return(x_left)

def calculate_x_right(da, config):
  N_x     = config.N_x
  N_ghost = config.N_ghost

  N_vel_x = config.N_vel_x
  N_vel_y = config.N_vel_y

  # Getting the step-size in x:
  x_start  = config.x_start
  x_end    = config.x_end
  length_x = x_end - x_start
  dx       = length_x/N_x

  # Obtaining the left-bottom corner coordinates 
  # of the left-bottom corner cell in the local zone considered:
  ((j_bottom, i_left), (N_y_local, N_x_local)) = da.getCorners()
  # Obtaining the right edge x-coordinate of the cell:
  i_right = i_left + 1

  # Constructing x_right using the above data:
  i       = i_right + np.arange(-N_ghost, N_x_local + N_ghost, 1)
  x_right = x_start + i * dx
  x_right = af.Array.as_type(af.to_array(x_right), af.Dtype.f64)

  # Reordering and tiling such that variation in x is along axis 1:
  x_right = af.tile(af.reorder(x_right), N_y_local + 2*N_ghost, 1, N_vel_y, N_vel_x)

  af.eval(x_right)
  return(x_right)

def calculate_x_center(da, config):
  N_x     = config.N_x
  N_ghost = config.N_ghost

  N_vel_x = config.N_vel_x
  N_vel_y = config.N_vel_y

  # Getting the step-size in x:
  x_start  = config.x_start
  x_end    = config.x_end
  length_x = x_end - x_start
  dx       = length_x/N_x

  # Obtaining the left-bottom corner coordinates 
  # of the left-bottom corner cell in the local zone considered:
  ((j_bottom, i_left), (N_y_local, N_x_local)) = da.getCorners()
  # Obtaining the cell-centered x-coordinate of the cell:
  i_center = i_left + 0.5

  # Constructing x_center using the above data:
  i        = i_center + np.arange(-N_ghost, N_x_local + N_ghost, 1)
  x_center = x_start  + i * dx
  x_center = af.Array.as_type(af.to_array(x_center), af.Dtype.f64)
  
  # Reordering and tiling such that variation in x is along axis 1:
  x_center = af.tile(af.reorder(x_center), N_y_local + 2*N_ghost, 1, N_vel_y, N_vel_x)

  af.eval(x_center)
  return(x_center)

def calculate_y_bottom(da, config):
  N_y     = config.N_y
  N_ghost = config.N_ghost
  
  N_vel_x = config.N_vel_x
  N_vel_y = config.N_vel_y

  # Getting the step-size in y:
  y_start  = config.y_start
  y_end    = config.y_end
  length_y = y_end - y_start
  dy       = length_y/N_y

  # Obtaining the left-bottom corner coordinates 
  # of the left-bottom corner cell in the local zone considered:
  ((j_bottom, i_left), (N_y_local, N_x_local)) = da.getCorners()

  # Constructing y_bottom using the above data:
  j        = j_bottom + np.arange(-N_ghost, N_y_local + N_ghost, 1)
  y_bottom = y_start  + j * dy
  y_bottom = af.Array.as_type(af.to_array(y_bottom), af.Dtype.f64)

  # Tiling such that variation in y is along axis 0:
  y_bottom = af.tile(y_bottom, 1, N_x_local + 2*N_ghost, N_vel_y, N_vel_x)

  af.eval(y_bottom)
  return(y_bottom)

def calculate_y_top(da, config):
  N_y     = config.N_y
  N_ghost = config.N_ghost
  
  N_vel_x = config.N_vel_x
  N_vel_y = config.N_vel_y

  # Getting the step-size in y:
  y_start  = config.y_start
  y_end    = config.y_end
  length_y = y_end - y_start
  dy       = length_y/N_y

  # Obtaining the left-bottom corner coordinates 
  # of the left-bottom corner cell in the local zone considered:
  ((j_bottom, i_left), (N_y_local, N_x_local)) = da.getCorners()
  # Obtaining the top edge y-coordinate of the cell:
  j_top = j_bottom + 1

  # Constructing y_top using the above data:
  j     = j_top + np.arange(-N_ghost, N_y_local + N_ghost, 1)
  y_top = y_start  + j * dy
  y_top = af.Array.as_type(af.to_array(y_top), af.Dtype.f64)

  # Tiling such that variation in y is along axis 0:
  y_top = af.tile(y_top, 1, N_x_local + 2*N_ghost, N_vel_y, N_vel_x)

  af.eval(y_top)
  return(y_top)

def calculate_y_center(da, config):
  N_y     = config.N_y
  N_ghost = config.N_ghost
  
  N_vel_x = config.N_vel_x
  N_vel_y = config.N_vel_y

  # Getting the step-size in y:
  y_start  = config.y_start
  y_end    = config.y_end
  length_y = y_end - y_start
  dy       = length_y/N_y

  # Obtaining the left-bottom corner coordinates 
  # of the left-bottom corner cell in the local zone considered:
  ((j_bottom, i_left), (N_y_local, N_x_local)) = da.getCorners()
  # Obtaining the cell-centered y-coordinate of the cell:
  j_center = j_bottom + 0.5

  # Constructing y_center using the above data:
  j        = j_center + np.arange(-N_ghost, N_y_local + N_ghost, 1)
  y_center = y_start  + j * dy
  y_center = af.Array.as_type(af.to_array(y_center), af.Dtype.f64)

  # Tiling such that variation in y is along axis 0:
  y_center = af.tile(y_center, 1, N_x_local + 2*N_ghost, N_vel_y, N_vel_x)

  af.eval(y_center)
  return(y_center)

def calculate_vel_x(da, config):

  N_vel_x = config.N_vel_x
  N_vel_y = config.N_vel_y
  N_ghost = config.N_ghost

  vel_x_max = config.vel_x_max

  # Obtaining the left-bottom corner coordinates 
  # of the left-bottom corner cell in the local zone considered:
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

  N_vel_x = config.N_vel_x
  N_vel_y = config.N_vel_y
  N_ghost = config.N_ghost

  vel_y_max = config.vel_y_max

  # Obtaining the left-bottom corner coordinates 
  # of the left-bottom corner cell in the local zone considered:
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

def f_initial(da, config):

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

  # Calculating x_center, y_center,vel_x and vel_y for the local zone:
  x_center = calculate_x_center(da, config)
  vel_x    = calculate_vel_x(da, config)
  y_center = calculate_y_center(da, config)
  vel_y    = calculate_vel_y(da, config)

  # Calculating the perturbed density:
  rho   = rho_background + (pert_real * af.cos(k_x*x_center + k_y*y_center) -\
                            pert_imag * af.sin(k_x*x_center + k_y*y_center)
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
    
  normalization = af.sum(f_initial) * dv_x * dv_y/(x_center.shape[0] * x_center.shape[1])
  f_initial     = f_initial #/normalization
  
  af.eval(f_initial)
  return(f_initial)