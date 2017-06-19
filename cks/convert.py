# Since we intend to use this code for a 2D3V simulation run, some manipulations
# need to be performed to allow us to run the same by making use of 4D array structures
# For this purpose, we define 2 forms for every array involved in the calculation:
# positionsExpanded  form : (Ny, Nx, Nvy*Nvx*Nvz, 1)
# velocitiesExpanded form : (Ny*Nx, Nvy, Nvx, Nvz, 1)

# This file contains the functions that will be used to convert
# from one array form to another

import arrayfire as af

def to_positionsExpanded(da, config, array):

  # Obtaining the left-bottom corner coordinates 
  # of the left-bottom corner cell in the local zone considered:
  ((j_bottom, i_left), (N_y_local, N_x_local)) = da.getCorners()

  array  = af.moddims(array,                   
                      (N_y_local + 2 * config.N_ghost),\
                      (N_x_local + 2 * config.N_ghost),\
                      config.N_vel_y*config.N_vel_x*config.N_vel_z,\
                      1
                     )
  af.eval(array)
  return(array)

def to_velocitiesExpanded(da, config, array):

  # Obtaining the left-bottom corner coordinates 
  # of the left-bottom corner cell in the local zone considered:
  ((j_bottom, i_left), (N_y_local, N_x_local)) = da.getCorners()
  
  array  = af.moddims(array,                   
                      (N_y_local + 2 * config.N_ghost)*\
                      (N_x_local + 2 * config.N_ghost),\
                      config.N_vel_y,\
                      config.N_vel_x,\
                      config.N_vel_z
                     ) 
  af.eval(array)
  return(array)