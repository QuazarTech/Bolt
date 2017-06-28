import numpy as np
import arrayfire as af 
import non_linear_solver.convert

def calculate_x_left(da, config):
  N_ghost = config.N_ghost

  N_vel_x = config.N_vel_x
  N_vel_y = config.N_vel_y
  N_vel_z = config.N_vel_z

  # Obtaining the left-bottom corner coordinates 
  # of the left-bottom corner cell in the local zone considered:
  ((j_bottom, i_left), (N_y_local, N_x_local)) = da.getCorners()

  # Constructing x_left using the above data:
  i      = i_left  + np.arange(-N_ghost, N_x_local + N_ghost, 1)
  x_left = config.x_start + i * config.dx
  x_left = af.Array.as_type(af.to_array(x_left), af.Dtype.f64)

  # Reordering and tiling such that variation in x is along axis 1:
  x_left = af.tile(af.reorder(x_left), N_y_local + 2*N_ghost, 1,\
                   N_vel_x*N_vel_y*N_vel_z, 1
                  )

  af.eval(x_left)
  # Returns in positionsExpanded form(Nx, Ny, Nvx*Nvy*Nvz, 1)
  return(x_left)

def calculate_x_right(da, config):
  N_ghost = config.N_ghost

  N_vel_x = config.N_vel_x
  N_vel_y = config.N_vel_y
  N_vel_z = config.N_vel_z

  # Obtaining the left-bottom corner coordinates 
  # of the left-bottom corner cell in the local zone considered:
  ((j_bottom, i_left), (N_y_local, N_x_local)) = da.getCorners()
  # Obtaining the right edge x-coordinate of the cell:
  i_right = i_left + 1

  # Constructing x_right using the above data:
  i       = i_right + np.arange(-N_ghost, N_x_local + N_ghost, 1)
  x_right = config.x_start + i * config.dx
  x_right = af.Array.as_type(af.to_array(x_right), af.Dtype.f64)

  # Reordering and tiling such that variation in x is along axis 1:
  x_right = af.tile(af.reorder(x_right), N_y_local + 2*N_ghost, 1,\
                    N_vel_x*N_vel_y*N_vel_z, 1
                   )

  af.eval(x_right)
  # Returns in positionsExpanded form(Nx, Ny, Nvx*Nvy*Nvz, 1)
  return(x_right)

def calculate_x_center(da, config):
  N_ghost = config.N_ghost

  N_vel_x = config.N_vel_x
  N_vel_y = config.N_vel_y
  N_vel_z = config.N_vel_z

  # Obtaining the left-bottom corner coordinates 
  # of the left-bottom corner cell in the local zone considered:
  ((j_bottom, i_left), (N_y_local, N_x_local)) = da.getCorners()
  # Obtaining the cell-centered x-coordinate of the cell:
  i_center = i_left + 0.5

  # Constructing x_center using the above data:
  i        = i_center + np.arange(-N_ghost, N_x_local + N_ghost, 1)
  x_center = config.x_start  + i * config.dx
  x_center = af.Array.as_type(af.to_array(x_center), af.Dtype.f64)
  
  # Reordering and tiling such that variation in x is along axis 1:
  x_center = af.tile(af.reorder(x_center), N_y_local + 2*N_ghost, 1,\
                     N_vel_x*N_vel_y*N_vel_z, 1
                    )

  af.eval(x_center)
  # Returns in positionsExpanded form(Nx, Ny, Nvx*Nvy*Nvz, 1)
  return(x_center)

def calculate_y_bottom(da, config):
  N_ghost = config.N_ghost
  
  N_vel_x = config.N_vel_x
  N_vel_y = config.N_vel_y
  N_vel_z = config.N_vel_z

  # Obtaining the left-bottom corner coordinates 
  # of the left-bottom corner cell in the local zone considered:
  ((j_bottom, i_left), (N_y_local, N_x_local)) = da.getCorners()

  # Constructing y_bottom using the above data:
  j        = j_bottom + np.arange(-N_ghost, N_y_local + N_ghost, 1)
  y_bottom = config.y_start  + j * config.dy
  y_bottom = af.Array.as_type(af.to_array(y_bottom), af.Dtype.f64)

  # Tiling such that variation in y is along axis 0:
  y_bottom = af.tile(y_bottom, 1, N_x_local + 2*N_ghost,\
                     N_vel_x*N_vel_y*N_vel_z, 1
                    )

  af.eval(y_bottom)
  # Returns in positionsExpanded form(Nx, Ny, Nvx*Nvy*Nvz, 1)
  return(y_bottom)

def calculate_y_top(da, config):
  N_ghost = config.N_ghost
  
  N_vel_x = config.N_vel_x
  N_vel_y = config.N_vel_y
  N_vel_z = config.N_vel_z

  # Obtaining the left-bottom corner coordinates 
  # of the left-bottom corner cell in the local zone considered:
  ((j_bottom, i_left), (N_y_local, N_x_local)) = da.getCorners()
  # Obtaining the top edge y-coordinate of the cell:
  j_top = j_bottom + 1

  # Constructing y_top using the above data:
  j     = j_top + np.arange(-N_ghost, N_y_local + N_ghost, 1)
  y_top = config.y_start  + j * config.dy
  y_top = af.Array.as_type(af.to_array(y_top), af.Dtype.f64)

  # Tiling such that variation in y is along axis 0:
  y_top = af.tile(y_top, 1, N_x_local + 2*N_ghost,\
                  N_vel_x*N_vel_y*N_vel_z, 1
                 )

  af.eval(y_top)
  # Returns in positionsExpanded form(Nx, Ny, Nvx*Nvy*Nvz, 1)
  return(y_top)

def calculate_y_center(da, config):
  N_ghost = config.N_ghost
  
  N_vel_x = config.N_vel_x
  N_vel_y = config.N_vel_y
  N_vel_z = config.N_vel_z

  # Obtaining the left-bottom corner coordinates 
  # of the left-bottom corner cell in the local zone considered:
  ((j_bottom, i_left), (N_y_local, N_x_local)) = da.getCorners()
  # Obtaining the cell-centered y-coordinate of the cell:
  j_center = j_bottom + 0.5

  # Constructing y_center using the above data:
  j        = j_center + np.arange(-N_ghost, N_y_local + N_ghost, 1)
  y_center = config.y_start  + j * config.dy
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

  # Obtaining the left-bottom corner coordinates 
  # of the left-bottom corner cell in the local zone considered:
  ((j_bottom, i_left), (N_y_local, N_x_local)) = da.getCorners()

  # These are the cell centered values in velocity space:
  # Constructing vel_x, vel_y and vel_z using the above data:
  i_v_x = 0.5 + np.arange(0, N_vel_x, 1)
  vel_x = -config.vel_x_max + i_v_x * config.dv_x
  vel_x = af.Array.as_type(af.to_array(vel_x), af.Dtype.f64)

  i_v_y = 0.5 + np.arange(0, N_vel_y, 1)
  vel_y = -config.vel_y_max + i_v_y * config.dv_y
  vel_y = af.Array.as_type(af.to_array(vel_y), af.Dtype.f64)

  i_v_z = 0.5 + np.arange(0, N_vel_z, 1)
  vel_z = -config.vel_z_max + i_v_z * config.dv_z
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

def f_initial(da, args):

  config             = args.config
  mass_particle      = config.mass_particle
  boltzmann_constant = config.boltzmann_constant

  rho_background         = config.rho_background
  temperature_background = config.temperature_background
  
  vel_bulk_x_background  = config.vel_bulk_x_background
  vel_bulk_y_background  = config.vel_bulk_y_background
  vel_bulk_z_background  = config.vel_bulk_z_background

  pert_real = config.pert_real
  pert_imag = config.pert_imag
 
  k_x = config.k_x
  k_y = config.k_y

  # Getting x, y, z at centers and vel_x,y,z for the local zone:
  x_center = args.x_center #positionsExpanded form
  y_center = args.y_center #positionsExpanded form

  vel_x, vel_y, vel_z = args.vel_x, args.vel_y, args.vel_z #velocitiesExpanded form

  # Calculating the perturbed density:
  rho = rho_background + (pert_real * af.cos(k_x*x_center + k_y*y_center) -\
                          pert_imag * af.sin(k_x*x_center + k_y*y_center)
                         )

  # Modifying the dimensions of rho:
  # Converting from positionsExpanded form to velocitiesExpanded form:
  rho = non_linear_solver.convert.to_velocitiesExpanded(da, config, rho)

  # Depending on the dimensionality in velocity space, the 
  # distribution function is assigned accordingly:
  if(config.mode == '3V'):
    
    args.f = rho * (mass_particle/(2*np.pi*boltzmann_constant*temperature_background))**(3/2) * \
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

    args.f = rho * (mass_particle/(2*np.pi*boltzmann_constant*temperature_background)) * \
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

    args.f = rho *\
             np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*temperature_background)) * \
             af.exp(-mass_particle*(vel_x - vel_bulk_x_background)**2/\
                   (2*boltzmann_constant*temperature_background))
    
    f_background = rho_background * \
                   np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*temperature_background)) * \
                   af.exp(-mass_particle*(vel_x - vel_bulk_x_background)**2/\
                         (2*boltzmann_constant*temperature_background))

    
  args.config.normalization = af.sum(f_background) * config.dv_x * config.dv_y * config.dv_z/\
                              (f_background.shape[0])
  args.f                    = af.log(args.f/args.config.normalization)
  
  # Modifying the dimensions again:
  # Converting from velocitiesExpanded form to positionsExpanded form:
  args.f = non_linear_solver.convert.to_positionsExpanded(da, config, args.f)

  af.eval(args.f)
  return(args)
  
def f_left(da, args):

  config             = args.config
  mass_particle      = config.mass_particle
  boltzmann_constant = config.boltzmann_constant

  left_rho         = config.left_rho
  left_temperature = config.left_temperature
  
  left_vel_bulk_x = config.left_vel_bulk_x
  left_vel_bulk_y = config.left_vel_bulk_y
  left_vel_bulk_z = config.left_vel_bulk_z

  vel_x, vel_y, vel_z = args.vel_x, args.vel_y, args.vel_z #velocitiesExpanded form

  # Depending on the dimensionality in velocity space, the 
  # distribution function is assigned accordingly:
  if(config.mode == '3V'):
    
    f_left = left_rho * (mass_particle/(2*np.pi*boltzmann_constant*left_temperature))**(3/2) * \
             af.exp(-mass_particle*(vel_x - left_vel_bulk_x)**2/\
                   (2*boltzmann_constant*left_temperature)) * \
             af.exp(-mass_particle*(vel_y - left_vel_bulk_y)**2/\
                   (2*boltzmann_constant*left_temperature)) * \
             af.exp(-mass_particle*(vel_z - left_vel_bulk_z)**2/\
                   (2*boltzmann_constant*left_temperature))

  elif(config.mode == '2V'):

    f_left = left_rho * (mass_particle/(2*np.pi*boltzmann_constant*left_temperature)) * \
             af.exp(-mass_particle*(vel_x - left_vel_bulk_x)**2/\
                   (2*boltzmann_constant*left_temperature)) * \
             af.exp(-mass_particle*(vel_y - left_vel_bulk_y)**2/\
                   (2*boltzmann_constant*left_temperature))

  else:

    f_left = left_rho *\
             np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*left_temperature)) * \
             af.exp(-mass_particle*(vel_x - left_vel_bulk_x)**2/\
                   (2*boltzmann_constant*left_temperature))

  f_left = f_left/config.normalization
  
  # Modifying the dimensions again:
  # Converting from velocitiesExpanded form to positionsExpanded form:
  f_left = non_linear_solver.convert.to_positionsExpanded(da, config, f_left)

  af.eval(f_left)
  return(f_left)

def f_right(da, args):

  config             = args.config
  mass_particle      = config.mass_particle
  boltzmann_constant = config.boltzmann_constant

  right_rho         = config.right_rho
  right_temperature = config.right_temperature
  
  right_vel_bulk_x = config.right_vel_bulk_x
  right_vel_bulk_y = config.right_vel_bulk_y
  right_vel_bulk_z = config.right_vel_bulk_z

  vel_x, vel_y, vel_z = args.vel_x, args.vel_y, args.vel_z #velocitiesExpanded form

  # Depending on the dimensionality in velocity space, the 
  # distribution function is assigned accordingly:
  if(config.mode == '3V'):
    
    f_right = right_rho * (mass_particle/(2*np.pi*boltzmann_constant*right_temperature))**(3/2) * \
              af.exp(-mass_particle*(vel_x - right_vel_bulk_x)**2/\
                    (2*boltzmann_constant*right_temperature)) * \
              af.exp(-mass_particle*(vel_y - right_vel_bulk_y)**2/\
                    (2*boltzmann_constant*right_temperature)) * \
              af.exp(-mass_particle*(vel_z - right_vel_bulk_z)**2/\
                    (2*boltzmann_constant*right_temperature))

  elif(config.mode == '2V'):

    f_right = right_rho * (mass_particle/(2*np.pi*boltzmann_constant*right_temperature)) * \
              af.exp(-mass_particle*(vel_x - right_vel_bulk_x)**2/\
                    (2*boltzmann_constant*right_temperature)) * \
              af.exp(-mass_particle*(vel_y - right_vel_bulk_y)**2/\
                    (2*boltzmann_constant*right_temperature))

  else:

    f_right = right_rho *\
              np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*right_temperature)) * \
              af.exp(-mass_particle*(vel_x - right_vel_bulk_x)**2/\
                    (2*boltzmann_constant*right_temperature))

  f_right = f_right/config.normalization
  
  # Modifying the dimensions again:
  # Converting from velocitiesExpanded form to positionsExpanded form:
  f_right = non_linear_solver.convert.to_positionsExpanded(da, config, f_right)

  af.eval(f_right)
  return(f_right)

def f_bot(da, args):

  config             = args.config
  mass_particle      = config.mass_particle
  boltzmann_constant = config.boltzmann_constant

  bot_rho         = config.bot_rho
  bot_temperature = config.bot_temperature
  
  bot_vel_bulk_x = config.bot_vel_bulk_x
  bot_vel_bulk_y = config.bot_vel_bulk_y
  bot_vel_bulk_z = config.bot_vel_bulk_z

  vel_x, vel_y, vel_z = args.vel_x, args.vel_y, args.vel_z #velocitiesExpanded form

  # Depending on the dimensionality in velocity space, the 
  # distribution function is assigned accordingly:
  if(config.mode == '3V'):
    
    f_bot = bot_rho * (mass_particle/(2*np.pi*boltzmann_constant*bot_temperature))**(3/2) * \
            af.exp(-mass_particle*(vel_x - bot_vel_bulk_x)**2/\
                  (2*boltzmann_constant*bot_temperature)) * \
            af.exp(-mass_particle*(vel_y - bot_vel_bulk_y)**2/\
                  (2*boltzmann_constant*bot_temperature)) * \
            af.exp(-mass_particle*(vel_z - bot_vel_bulk_z)**2/\
                  (2*boltzmann_constant*bot_temperature))

  elif(config.mode == '2V'):

    f_bot = bot_rho * (mass_particle/(2*np.pi*boltzmann_constant*bot_temperature)) * \
            af.exp(-mass_particle*(vel_x - bot_vel_bulk_x)**2/\
                  (2*boltzmann_constant*bot_temperature)) * \
            af.exp(-mass_particle*(vel_y - bot_vel_bulk_y)**2/\
                  (2*boltzmann_constant*bot_temperature))

  else:

    f_bot = bot_rho *\
            np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*bot_temperature)) * \
            af.exp(-mass_particle*(vel_x - bot_vel_bulk_x)**2/\
                  (2*boltzmann_constant*bot_temperature))

  f_bot = f_bot/config.normalization
  
  # Modifying the dimensions again:
  # Converting from velocitiesExpanded form to positionsExpanded form:
  f_bot = non_linear_solver.convert.to_positionsExpanded(da, config, f_bot)

  af.eval(f_bot)
  return(f_bot)

def f_top(da, args):

  config             = args.config
  mass_particle      = config.mass_particle
  boltzmann_constant = config.boltzmann_constant

  top_rho         = config.top_rho
  top_temperature = config.top_temperature
  
  top_vel_bulk_x = config.top_vel_bulk_x
  top_vel_bulk_y = config.top_vel_bulk_y
  top_vel_bulk_z = config.top_vel_bulk_z

  vel_x, vel_y, vel_z = args.vel_x, args.vel_y, args.vel_z #velocitiesExpanded form

  # Depending on the dimensionality in velocity space, the 
  # distribution function is assigned accordingly:
  if(config.mode == '3V'):
    
    f_top = top_rho * (mass_particle/(2*np.pi*boltzmann_constant*top_temperature))**(3/2) * \
            af.exp(-mass_particle*(vel_x - top_vel_bulk_x)**2/\
                  (2*boltzmann_constant*top_temperature)) * \
            af.exp(-mass_particle*(vel_y - top_vel_bulk_y)**2/\
                  (2*boltzmann_constant*top_temperature)) * \
            af.exp(-mass_particle*(vel_z - top_vel_bulk_z)**2/\
                  (2*boltzmann_constant*top_temperature))

  elif(config.mode == '2V'):

    f_top = top_rho * (mass_particle/(2*np.pi*boltzmann_constant*top_temperature)) * \
            af.exp(-mass_particle*(vel_x - top_vel_bulk_x)**2/\
                  (2*boltzmann_constant*top_temperature)) * \
            af.exp(-mass_particle*(vel_y - top_vel_bulk_y)**2/\
                  (2*boltzmann_constant*top_temperature))

  else:

    f_top = top_rho *\
            np.sqrt(mass_particle/(2*np.pi*boltzmann_constant*top_temperature)) * \
            af.exp(-mass_particle*(vel_x - top_vel_bulk_x)**2/\
                  (2*boltzmann_constant*top_temperature))

  f_top = f_top/config.normalization
  
  # Modifying the dimensions again:
  # Converting from velocitiesExpanded form to positionsExpanded form:
  f_top = non_linear_solver.convert.to_positionsExpanded(da, config, f_top)

  af.eval(f_top)
  return(f_top)