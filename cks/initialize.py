import numpy as np
import arrayfire as af 
import cks.convert

def calculate_x_left(da, config):
  N_x     = config.N_x
  N_ghost = config.N_ghost

  N_vel_x = config.N_vel_x
  N_vel_y = config.N_vel_y
  N_vel_z = config.N_vel_z

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
  x_left = af.tile(af.reorder(x_left), N_y_local + 2*N_ghost, 1,\
                   N_vel_x*N_vel_y*N_vel_z, 1
                  )

  af.eval(x_left)
  # Returns in positionsExpanded form(Nx, Ny, Nvx*Nvy*Nvz, 1)
  return(x_left)

def calculate_x_right(da, config):
  N_x     = config.N_x
  N_ghost = config.N_ghost

  N_vel_x = config.N_vel_x
  N_vel_y = config.N_vel_y
  N_vel_z = config.N_vel_z

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
  x_right = af.tile(af.reorder(x_right), N_y_local + 2*N_ghost, 1,\
                    N_vel_x*N_vel_y*N_vel_z, 1
                   )

  af.eval(x_right)
  # Returns in positionsExpanded form(Nx, Ny, Nvx*Nvy*Nvz, 1)
  return(x_right)

def calculate_x_center(da, config):
  N_x     = config.N_x
  N_ghost = config.N_ghost

  N_vel_x = config.N_vel_x
  N_vel_y = config.N_vel_y
  N_vel_z = config.N_vel_z

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
  x_center = af.tile(af.reorder(x_center), N_y_local + 2*N_ghost, 1,\
                     N_vel_x*N_vel_y*N_vel_z, 1
                    )

  af.eval(x_center)
  # Returns in positionsExpanded form(Nx, Ny, Nvx*Nvy*Nvz, 1)
  return(x_center)

def calculate_y_bottom(da, config):
  N_y     = config.N_y
  N_ghost = config.N_ghost
  
  N_vel_x = config.N_vel_x
  N_vel_y = config.N_vel_y
  N_vel_z = config.N_vel_z

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
  y_bottom = af.tile(y_bottom, 1, N_x_local + 2*N_ghost,\
                     N_vel_x*N_vel_y*N_vel_z, 1
                    )

  af.eval(y_bottom)
  # Returns in positionsExpanded form(Nx, Ny, Nvx*Nvy*Nvz, 1)
  return(y_bottom)

def calculate_y_top(da, config):
  N_y     = config.N_y
  N_ghost = config.N_ghost
  
  N_vel_x = config.N_vel_x
  N_vel_y = config.N_vel_y
  N_vel_z = config.N_vel_z

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
  y_top = af.tile(y_top, 1, N_x_local + 2*N_ghost,\
                  N_vel_x*N_vel_y*N_vel_z, 1
                 )

  af.eval(y_top)
  # Returns in positionsExpanded form(Nx, Ny, Nvx*Nvy*Nvz, 1)
  return(y_top)

def calculate_y_center(da, config):
  N_y     = config.N_y
  N_ghost = config.N_ghost
  
  N_vel_x = config.N_vel_x
  N_vel_y = config.N_vel_y
  N_vel_z = config.N_vel_z

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
  y_center = af.tile(y_center, 1, N_x_local + 2*N_ghost,\
                     N_vel_x*N_vel_y*N_vel_z, 1
                    )

  af.eval(y_center)
  # Returns in positionsExpanded form(Nx, Ny, Nvx*Nvy*Nvz, 1)
  return(y_center)

def calculate_velocities(da, config):

  N_vel_x = config.N_vel_x
  N_vel_y = config.N_vel_y
  N_vel_z = config.N_vel_z
  
  N_ghost = config.N_ghost

  vel_x_max = config.vel_x_max
  vel_y_max = config.vel_y_max
  vel_z_max = config.vel_z_max

  # Obtaining the left-bottom corner coordinates 
  # of the left-bottom corner cell in the local zone considered:
  ((j_bottom, i_left), (N_y_local, N_x_local)) = da.getCorners()

  # These are the cell centered values in velocity space:
  # Constructing vel_x, vel_y and vel_z using the above data:
  i_v_x = 0.5 + np.arange(0, N_vel_x, 1)
  dv_x  = (2*vel_x_max)/N_vel_x
  vel_x = -vel_x_max + i_v_x * dv_x
  vel_x = af.Array.as_type(af.to_array(vel_x), af.Dtype.f64)

  i_v_y = 0.5 + np.arange(0, N_vel_y, 1)
  dv_y  = (2*vel_y_max)/N_vel_y
  vel_y = -vel_y_max + i_v_y * dv_y
  vel_y = af.Array.as_type(af.to_array(vel_y), af.Dtype.f64)

  i_v_z = 0.5 + np.arange(0, N_vel_z, 1)
  dv_z  = (2*vel_z_max)/N_vel_z
  vel_z = -vel_z_max + i_v_z * dv_z
  vel_z = af.Array.as_type(af.to_array(vel_z), af.Dtype.f64)

  # Reordering and tiling such that variation in x-velocity is along axis 2
  vel_x = af.reorder(vel_x, 2, 3, 0, 1)
  vel_x = af.tile(vel_x,\
                  (N_x_local + 2 * N_ghost)*\
                  (N_y_local + 2 * N_ghost),\
                  N_vel_y, \
                  1, \
                  N_vel_z
                 )

  # Reordering and tiling such that variation in y-velocity is along axis 1
  vel_y = af.reorder(vel_y, 3, 0, 1, 2)
  vel_y = af.tile(vel_y,\
                  (N_x_local + 2 * N_ghost)*\
                  (N_y_local + 2 * N_ghost),\
                  1, \
                  N_vel_x, \
                  N_vel_z
                 )

  # Reordering and tiling such that variation in y-velocity is along axis 3
  vel_z = af.reorder(vel_z, 1, 2, 3, 0)
  vel_z = af.tile(vel_z,\
                  (N_x_local + 2 * N_ghost)*\
                  (N_y_local + 2 * N_ghost),\
                  N_vel_y, \
                  N_vel_x, \
                  1
                 )

  af.eval(vel_x, vel_y, vel_z)
  # Returns in velocitiesExpanded form(Nx*Ny, Nvx, Nvy, Nvz)
  return(vel_x, vel_y, vel_z)

def f_initial(da, config):

  mass_particle      = config.mass_particle
  boltzmann_constant = config.boltzmann_constant

  rho_background         = config.rho_background
  temperature_background = config.temperature_background
  
  vel_bulk_x_background  = config.vel_bulk_x_background
  vel_bulk_y_background  = config.vel_bulk_y_background
  vel_bulk_z_background  = config.vel_bulk_z_background

  N_vel_x = config.N_vel_x
  N_vel_y = config.N_vel_y
  N_vel_z = config.N_vel_z

  vel_y_max = config.vel_y_max
  vel_x_max = config.vel_x_max
  vel_z_max = config.vel_z_max

  dv_x = (2*vel_x_max)/N_vel_x
  dv_y = (2*vel_y_max)/N_vel_y
  dv_z = (2*vel_z_max)/N_vel_z

  pert_real = config.pert_real
  pert_imag = config.pert_imag
 
  k_x = config.k_x
  k_y = config.k_y

  # Calculating x, y, z at centers and vel_x,y,z for the local zone:
  x_center = calculate_x_center(da, config) #positionsExpanded form
  y_center = calculate_y_center(da, config) #positionsExpanded form

  vel_x, vel_y, vel_z = calculate_velocities(da, config) #velocitiesExpanded form

  # Obtaining the left-bottom corner coordinates 
  # of the left-bottom corner cell in the local zone considered:
  ((j_bottom, i_left), (N_y_local, N_x_local)) = da.getCorners()

  # Calculating the perturbed density:
  rho = rho_background + (pert_real * af.cos(k_x*x_center + k_y*y_center) -\
                          pert_imag * af.sin(k_x*x_center + k_y*y_center)
                         )

  # Modifying the dimensions of rho:
  # Converting from positionsExpanded form to velocitiesExpanded form:
  rho = cks.convert.to_velocitiesExpanded(da, config, rho)

  # Depending on the dimensionality in velocity space, the 
  # distribution function is assigned accordingly:
  if(config.mode == '3V'):
    
    f_initial = rho * (mass_particle/(2*np.pi*boltzmann_constant*temperature_background))**(3/2) * \
                af.exp(-mass_particle*(vel_x - vel_bulk_x_background)**2/\
                      (2*boltzmann_constant*temperature_background)) * \
                af.exp(-mass_particle*(vel_y - vel_bulk_y_background)**2/\
                      (2*boltzmann_constant*temperature_background)) * \
                af.exp(-mass_particle*(vel_z - vel_bulk_z_background)**2/\
                      (2*boltzmann_constant*temperature_background))

    f_background = rho_background * \
                   (mass_particle/(2*np.pi*boltzmann_constant*temperature_background))**(3/2) * \
                    af.exp(-mass_particle*(vel_x - vel_bulk_x_background)**2/\
                          (2*boltzmann_constant*temperature_background)) * \
                    af.exp(-mass_particle*(vel_y - vel_bulk_x_background)**2/\
                          (2*boltzmann_constant*temperature_background)) * \
                    af.exp(-mass_particle*(vel_z - vel_bulk_z_background)**2/\
                          (2*boltzmann_constant*temperature_background))

  elif(config.mode == '2V'):

    f_initial = rho * (mass_particle/(2*np.pi*boltzmann_constant*temperature_background)) * \
                af.exp(-mass_particle*(vel_x - vel_bulk_x_background)**2/\
                      (2*boltzmann_constant*temperature_background)) * \
                af.exp(-mass_particle*(vel_y - vel_bulk_y_background)**2/\
                      (2*boltzmann_constant*temperature_background))

    f_background = rho_background * \
                   (mass_particle/(2*np.pi*boltzmann_constant*temperature_background)) * \
                    af.exp(-mass_particle*(vel_x - vel_bulk_x_background)**2/\
                          (2*boltzmann_constant*temperature_background)) * \
                    af.exp(-mass_particle*(vel_y - vel_bulk_x_background)**2/\
                          (2*boltzmann_constant*temperature_background))


  else:

    f_initial = rho *\
                np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*temperature_background)) * \
                af.exp(-mass_particle*(vel_x - vel_bulk_x_background)**2/\
                      (2*boltzmann_constant*temperature_background))

    f_background = rho_background * \
                   np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*temperature_background)) * \
                   af.exp(-mass_particle*(vel_x - vel_bulk_x_background)**2/\
                         (2*boltzmann_constant*temperature_background))

    
  normalization = af.sum(f_background) * dv_x * dv_y * dv_z/(f_background.shape[0])
  f_initial     = f_initial/normalization
  
  # Modifying the dimensions again:
  # Converting from velocitiesExpanded form to positionsExpanded form:
  f_initial = cks.convert.to_positionsExpanded(da, config, f_initial)

  af.eval(f_initial)
  return(f_initial)