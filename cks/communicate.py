# This file contains the functions that are used to take care of the interzonal
# communications when the code is run in parallel across multiple nodes. 
# Additionally, these functions are also responsible for applying boundary conditions.

import numpy as np
import arrayfire as af

def communicate_distribution_function(da, args, local, glob):

  # Accessing the values of the global and local Vectors
  local_value = da.getVecArray(local)
  glob_value  = da.getVecArray(glob)

  N_ghost = args.config.N_ghost

  # Applying the boundary conditions:
  args.f         = apply_BC_distribution_function(da, args)

  # Storing values of af.Array in PETSc.Vec:
  local_value[:] = np.array(args.f)
  
  # Global value is non-inclusive of the ghost-zones:
  glob_value[:] = (local_value[:])[N_ghost:-N_ghost,\
                                   N_ghost:-N_ghost,\
                                   :
                                  ]

  # The following function takes care of the boundary conditions, 
  # and interzonal communications:
  da.globalToLocal(glob, local)

  # Converting back from PETSc.Vec to af.Array:
  f_updated = af.to_array(local_value[:])

  af.eval(f_updated)
  return(f_updated)

def apply_BC_distribution_function(da, args):

  config  = args.config
  N_ghost = config.N_ghost
  vel_x   = args.vel_x
  vel_y   = args.vel_y
  vel_z   = args.vel_z

  mass_particle      = config.mass_particle
  boltzmann_constant = config.boltzmann_constant

  # Obtaining the left-bottom corner coordinates 
  # of the left-bottom corner cell in the local zone considered:
  ((j_bottom, i_left), (N_y_local, N_x_local)) = da.getCorners()

  (j_top, i_right) = (j_bottom + 1, i_left + 1)

  if(args.config.bc_in_x == 'dirichlet'):
    left_rho         = config.left_rho           
    left_temperature = config.left_temperature 
    left_vel_bulk_x  = config.left_vel_bulk_x  
    left_vel_bulk_y  = config.left_vel_bulk_y
    left_vel_bulk_z  = config.left_vel_bulk_z

    right_rho         = config.right_rho           
    right_temperature = config.right_temperature 
    right_vel_bulk_x  = config.right_vel_bulk_x  
    right_vel_bulk_y  = config.right_vel_bulk_y
    right_vel_bulk_z  = config.right_vel_bulk_z

    if(config.mode == '3V'):
      f_left = left_rho * (mass_particle/(2*np.pi*boltzmann_constant*left_temperature))**(3/2) * \
               af.exp(-mass_particle*(vel_x[:, :N_ghost] - left_vel_bulk_x)**2/\
                     (2*boltzmann_constant*left_temperature)) * \
               af.exp(-mass_particle*(vel_y[:, :N_ghost] - left_vel_bulk_y)**2/\
                     (2*boltzmann_constant*left_temperature)) * \
               af.exp(-mass_particle*(vel_z[:, :N_ghost] - left_vel_bulk_z)**2/\
                     (2*boltzmann_constant*left_temperature))

      f_right = right_rho * (mass_particle/(2*np.pi*boltzmann_constant*right_temperature))**(3/2) * \
                af.exp(-mass_particle*(vel_x[:, -N_ghost:] - right_vel_bulk_x)**2/\
                      (2*boltzmann_constant*right_temperature)) * \
                af.exp(-mass_particle*(vel_y[:, -N_ghost:] - right_vel_bulk_y)**2/\
                      (2*boltzmann_constant*right_temperature)) * \
                af.exp(-mass_particle*(vel_z[:, -N_ghost:] - right_vel_bulk_z)**2/\
                      (2*boltzmann_constant*right_temperature))
    
    elif(config.mode == '2V'):
      f_left = left_rho * (mass_particle/(2*np.pi*boltzmann_constant*left_temperature)) * \
               af.exp(-mass_particle*(vel_x[:, :N_ghost] - left_vel_bulk_x)**2/\
                     (2*boltzmann_constant*left_temperature)) * \
               af.exp(-mass_particle*(vel_y[:, :N_ghost] - left_vel_bulk_y)**2/\
                     (2*boltzmann_constant*left_temperature))

      f_right = right_rho * (mass_particle/(2*np.pi*boltzmann_constant*right_temperature)) * \
                af.exp(-mass_particle*(vel_x[:, -N_ghost:] - right_vel_bulk_x)**2/\
                      (2*boltzmann_constant*right_temperature)) * \
                af.exp(-mass_particle*(vel_y[:, -N_ghost:] - right_vel_bulk_y)**2/\
                      (2*boltzmann_constant*right_temperature))

    else:
      f_left = left_rho * (mass_particle/(2*np.pi*boltzmann_constant*left_temperature))**(1/2) * \
               af.exp(-mass_particle*(vel_x[:, :N_ghost] - left_vel_bulk_x)**2/\
                     (2*boltzmann_constant*left_temperature))

      f_right = right_rho * (mass_particle/(2*np.pi*boltzmann_constant*right_temperature))**(1/2) * \
                af.exp(-mass_particle*(vel_x[:, -N_ghost:] - right_vel_bulk_x)**2/\
                      (2*boltzmann_constant*right_temperature))

    if(i_left == 0):
      args.f[:, :N_ghost] = f_left

    if(i_right == config.N_x - 1):
      args.f[:, -N_ghost:] = f_right

  if(args.config.bc_in_y == 'dirichlet'):
    bot_rho         = config.bot_rho           
    bot_temperature = config.bot_temperature 
    bot_vel_bulk_x  = config.bot_vel_bulk_x  
    bot_vel_bulk_y  = config.bot_vel_bulk_y
    bot_vel_bulk_z  = config.bot_vel_bulk_z

    top_rho         = config.top_rho           
    top_temperature = config.top_temperature 
    top_vel_bulk_x  = config.top_vel_bulk_x  
    top_vel_bulk_y  = config.top_vel_bulk_y
    top_vel_bulk_z  = config.top_vel_bulk_z

    if(config.mode == '3V'):
      f_bot = bot_rho * (mass_particle/(2*np.pi*boltzmann_constant*bot_temperature))**(3/2) * \
              af.exp(-mass_particle*(vel_x[:, :N_ghost] - bot_vel_bulk_x)**2/\
                    (2*boltzmann_constant*bot_temperature)) * \
              af.exp(-mass_particle*(vel_y[:, :N_ghost] - bot_vel_bulk_y)**2/\
                    (2*boltzmann_constant*bot_temperature)) * \
              af.exp(-mass_particle*(vel_z[:, :N_ghost] - bot_vel_bulk_z)**2/\
                    (2*boltzmann_constant*bot_temperature))

      f_top = top_rho * (mass_particle/(2*np.pi*boltzmann_constant*top_temperature))**(3/2) * \
              af.exp(-mass_particle*(vel_x[:, -N_ghost:] - top_vel_bulk_x)**2/\
                    (2*boltzmann_constant*top_temperature)) * \
              af.exp(-mass_particle*(vel_y[:, -N_ghost:] - top_vel_bulk_y)**2/\
                    (2*boltzmann_constant*top_temperature)) * \
              af.exp(-mass_particle*(vel_z[:, -N_ghost:] - top_vel_bulk_z)**2/\
                    (2*boltzmann_constant*top_temperature))
    
    elif(config.mode == '2V'):
      f_bot = bot_rho * (mass_particle/(2*np.pi*boltzmann_constant*bot_temperature)) * \
              af.exp(-mass_particle*(vel_x[:, :N_ghost] - bot_vel_bulk_x)**2/\
                    (2*boltzmann_constant*bot_temperature)) * \
              af.exp(-mass_particle*(vel_y[:, :N_ghost] - bot_vel_bulk_y)**2/\
                    (2*boltzmann_constant*bot_temperature))

      f_top = top_rho * (mass_particle/(2*np.pi*boltzmann_constant*top_temperature)) * \
              af.exp(-mass_particle*(vel_x[:, -N_ghost:] - top_vel_bulk_x)**2/\
                    (2*boltzmann_constant*top_temperature)) * \
              af.exp(-mass_particle*(vel_y[:, -N_ghost:] - top_vel_bulk_y)**2/\
                    (2*boltzmann_constant*top_temperature))

    else:
      f_bot = bot_rho * (mass_particle/(2*np.pi*boltzmann_constant*bot_temperature))**(1/2) * \
              af.exp(-mass_particle*(vel_x[:, :N_ghost] - bot_vel_bulk_x)**2/\
                    (2*boltzmann_constant*bot_temperature))

      f_top = top_rho * (mass_particle/(2*np.pi*boltzmann_constant*top_temperature))**(1/2) * \
              af.exp(-mass_particle*(vel_x[:, -N_ghost:] - top_vel_bulk_x)**2/\
                    (2*boltzmann_constant*top_temperature))

    if(j_bottom == 0):
      args.f[:N_ghost, :] = f_bot

    if(j_top == config.N_y - 1):
      args.f[-N_ghost:, :] = f_top

  af.eval(args.f)
  return(args.f)

def communicate_fields(da, args, local, glob):

  # Accessing the values of the global and local Vectors
  local_value = da.getVecArray(local)
  glob_value  = da.getVecArray(glob)

  N_ghost = args.config.N_ghost

  # Assigning the values of the af.Array fields quantities
  # to the PETSc.Vec:
  (local_value[:])[:, :, 0] = np.array(args.E_x)
  (local_value[:])[:, :, 1] = np.array(args.E_y)
  (local_value[:])[:, :, 2] = np.array(args.E_z)
  
  (local_value[:])[:, :, 3] = np.array(args.B_x)
  (local_value[:])[:, :, 4] = np.array(args.B_y)
  (local_value[:])[:, :, 5] = np.array(args.B_z)

  # Global value is non-inclusive of the ghost-zones:
  glob_value[:] = (local_value[:])[N_ghost:-N_ghost,\
                                   N_ghost:-N_ghost,\
                                   :
                                  ]

  # Takes care of boundary conditions and interzonal communications:
  da.globalToLocal(glob, local)

  # Converting back to af.Array
  args.E_x = af.to_array((local_value[:])[:, :, 0])
  args.E_y = af.to_array((local_value[:])[:, :, 1])
  args.E_z = af.to_array((local_value[:])[:, :, 2])

  args.B_x = af.to_array((local_value[:])[:, :, 3])
  args.B_y = af.to_array((local_value[:])[:, :, 4])
  args.B_z = af.to_array((local_value[:])[:, :, 5])

  return(args)