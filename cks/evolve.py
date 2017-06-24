import numpy as np
import arrayfire as af

import cks.convert
import cks.compute_moments
import cks.communicate

# Importing interpolation routines:
from cks.interpolation_routines import f_interp_2d, f_interp_vel_3d

# Importing the fields solvers:
from cks.EM_fields_solver.electrostatic import solve_electrostatic_fields
from cks.EM_fields_solver.fdtd import fdtd, fdtd_grid_to_ck_grid

# Importing the collision operators:
from cks.collision_operators.BGK import collision_step_BGK

def fields_step(da, args, local, glob, dt, flag = 0):

  config  = args.config
  N_ghost = config.N_ghost

  charge_electron = config.charge_electron

  # Obtaining the left-bottom corner coordinates 
  # of the left-bottom corner cell in the local zone considered:
  ((j_bottom, i_left), (N_y_local, N_x_local)) = da.getCorners()

  vel_x = args.vel_x
  vel_y = args.vel_y
  vel_z = args.vel_z

  # Convert to velocitiesExpanded:
  args.f      = cks.convert.to_velocitiesExpanded(da, config, args.f)
  args.f_half = cks.convert.to_velocitiesExpanded(da, config, args.f_half)

  # if(config.fields_solver == 'electrostatic'):
  #   E_x = af.constant(0, N_y_local + 2*N_ghost, N_x_local + 2*N_ghost, dtype = af.Dtype.f64)
  #   E_y = af.constant(0, N_y_local + 2*N_ghost, N_x_local + 2*N_ghost, dtype = af.Dtype.f64)
  #   E_z = af.constant(0, N_y_local + 2*N_ghost, N_x_local + 2*N_ghost, dtype = af.Dtype.f64)
    
  #   B_x = af.constant(0, N_y_local + 2*N_ghost, N_x_local + 2*N_ghost, dtype = af.Dtype.f64)
  #   B_y = af.constant(0, N_y_local + 2*N_ghost, N_x_local + 2*N_ghost, dtype = af.Dtype.f64)
  #   B_z = af.constant(0, N_y_local + 2*N_ghost, N_x_local + 2*N_ghost, dtype = af.Dtype.f64)

  #   rho_array = charge_electron * (cks.compute_moments.calculate_density(args) - \
  #                                  config.rho_background
  #                                 )#(i + 1/2, j + 1/2)
    
  #   # Passing the values non-inclusive of the ghost zones:
  #   rho_array = af.moddims(rho_array,\
  #                          N_y_local + 2 * N_ghost,\
  #                          N_x_local + 2 * N_ghost
  #                         )

  #   rho_array = np.array(rho_array)[N_ghost:-N_ghost,\
  #                                   N_ghost:-N_ghost
  #                                  ]
    
  #   E_x, E_y =\
  #   solve_electrostatic_fields(da, config, rho_array)

  # else:
  #   # Will returned a flattened array containing the values of J_x,y,z in 2D space:
  #   args.J_x = charge_electron * cks.compute_moments.calculate_mom_bulk_x(args) #(i + 1/2, j + 1/2)
  #   args.J_y = charge_electron * cks.compute_moments.calculate_mom_bulk_y(args) #(i + 1/2, j + 1/2)
  #   args.J_z = charge_electron * cks.compute_moments.calculate_mom_bulk_z(args) #(i + 1/2, j + 1/2)

  #   # We'll convert these back to 2D arrays to be able to perform FDTD:
  #   args.J_x = af.moddims(args.J_x,\
  #                         N_y_local + 2 * N_ghost,\
  #                         N_x_local + 2 * N_ghost
  #                        )
    
  #   args.J_y = af.moddims(args.J_y,\
  #                         N_y_local + 2 * N_ghost,\
  #                         N_x_local + 2 * N_ghost
  #                        )

  #   args.J_z = af.moddims(args.J_z,\
  #                         N_y_local + 2 * N_ghost,\
  #                         N_x_local + 2 * N_ghost
  #                        )

  #   # Obtaining the values for current density on the Yee-Grid:
  #   args.J_x = 0.5 * (args.J_x + af.shift(args.J_x, 1, 0)) #(i + 1/2, j)
  #   args.J_y = 0.5 * (args.J_y + af.shift(args.J_y, 0, 1)) #(i, j + 1/2)
  #   args.J_z = 0.25 * (args.J_z + af.shift(args.J_z, 1, 0) +\
  #                      af.shift(args.J_z, 0, 1) + af.shift(args.J_z, 1, 1)
  #                     ) #(i, j)

  #   # Storing the values for the previous half-time step:
  #   # We do this since the B values on the CK grid are defined at time t = n
  #   # While the B values on the FDTD grid are defined at t = n - 1/2
  #   # B_x_old, B_y_old, B_z_old = args.B_x.copy(), args.B_y.copy(), args.B_z.copy()
  #   # args = fdtd(da, args, local, glob, dt)
  
  #   # To account for half-time steps:
  #   # B_x = 0.5 * (args.B_x + B_x_old)
  #   # B_y = 0.5 * (args.B_y + B_y_old)
  #   # B_z = 0.5 * (args.B_z + B_z_old)
    
  #   # E_x, E_y, E_z, B_x, B_y, B_z = fdtd_grid_to_ck_grid(args.E_x, args.E_y, args.E_z,\
  #                                                       # B_x, B_y, B_z
  #                                                      # )
  #   E_x = args.E_x
  #   E_y = args.E_y
  #   E_z = args.E_z
    
  #   B_x = args.B_x
  #   B_y = args.B_y
  #   B_z = args.B_z
                                             
  # # Tiling such that E_x, E_y and B_z have the same array dimensions as f:
  # # This is required to perform the interpolation in velocity space:
  # # NOTE: Here we are making the assumption that when mode == '2V'/'1V', N_vel_z = 1
  # # If otherwise code will break here.
  # if(config.mode == '3V'):
  #   E_x = af.tile(af.flat(E_x), 1, args.f.shape[1], args.f.shape[2], args.f.shape[3]) #(i + 1/2, j + 1/2)
  #   E_y = af.tile(af.flat(E_y), 1, args.f.shape[1], args.f.shape[2], args.f.shape[3]) #(i + 1/2, j + 1/2)
  #   E_z = af.tile(af.flat(E_z), 1, args.f.shape[1], args.f.shape[2], args.f.shape[3]) #(i + 1/2, j + 1/2)

  #   B_x = af.tile(af.flat(B_x), 1, args.f.shape[1], args.f.shape[2], args.f.shape[3]) #(i + 1/2, j + 1/2)
  #   B_y = af.tile(af.flat(B_y), 1, args.f.shape[1], args.f.shape[2], args.f.shape[3]) #(i + 1/2, j + 1/2)
  #   B_z = af.tile(af.flat(B_z), 1, args.f.shape[1], args.f.shape[2], args.f.shape[3]) #(i + 1/2, j + 1/2)
 
  # else:
  #   E_x = af.tile(af.flat(E_x), 1, args.f.shape[1], args.f.shape[2], 1) #(i + 1/2, j + 1/2)
  #   E_y = af.tile(af.flat(E_y), 1, args.f.shape[1], args.f.shape[2], 1) #(i + 1/2, j + 1/2)
  #   E_z = af.tile(af.flat(E_z), 1, args.f.shape[1], args.f.shape[2], 1) #(i + 1/2, j + 1/2)

  #   B_x = af.tile(af.flat(B_x), 1, args.f.shape[1], args.f.shape[2], 1) #(i + 1/2, j + 1/2)
  #   B_y = af.tile(af.flat(B_y), 1, args.f.shape[1], args.f.shape[2], 1) #(i + 1/2, j + 1/2)
  #   B_z = af.tile(af.flat(B_z), 1, args.f.shape[1], args.f.shape[2], 1) #(i + 1/2, j + 1/2)

  # F_x = charge_electron * (E_x + vel_y * B_z - vel_z * B_y) #(i + 1/2, j + 1/2)
  # F_y = charge_electron * (E_y - vel_x * B_z + vel_z * B_x) #(i + 1/2, j + 1/2)
  # F_z = charge_electron * (E_z - vel_y * B_x + vel_x * B_y) #(i + 1/2, j + 1/2)

  # args.f_half = f_interp_vel_3d(args, F_x, F_y, F_z, dt/2, 1)

  if(config.fields_solver == 'electrostatic'):
    E_x = af.constant(0, N_y_local + 2*N_ghost, N_x_local + 2*N_ghost, dtype = af.Dtype.f64)
    E_y = af.constant(0, N_y_local + 2*N_ghost, N_x_local + 2*N_ghost, dtype = af.Dtype.f64)
    E_z = af.constant(0, N_y_local + 2*N_ghost, N_x_local + 2*N_ghost, dtype = af.Dtype.f64)
    
    B_x = af.constant(0, N_y_local + 2*N_ghost, N_x_local + 2*N_ghost, dtype = af.Dtype.f64)
    B_y = af.constant(0, N_y_local + 2*N_ghost, N_x_local + 2*N_ghost, dtype = af.Dtype.f64)
    B_z = af.constant(0, N_y_local + 2*N_ghost, N_x_local + 2*N_ghost, dtype = af.Dtype.f64)

    rho_array = charge_electron * (cks.compute_moments.calculate_density(args, 1) - \
                                   config.rho_background
                                  )#(i + 1/2, j + 1/2)
    
    # Passing the values non-inclusive of the ghost zones:
    rho_array = af.moddims(rho_array,\
                           N_y_local + 2 * N_ghost,\
                           N_x_local + 2 * N_ghost
                          )

    rho_array = np.array(rho_array)[N_ghost:-N_ghost,\
                                    N_ghost:-N_ghost
                                   ]
    
    E_x, E_y =\
    solve_electrostatic_fields(da, config, rho_array)
 
  if(config.mode == '3V'):
    E_x = af.tile(af.flat(E_x), 1, args.f.shape[1], args.f.shape[2], args.f.shape[3]) #(i + 1/2, j + 1/2)
    E_y = af.tile(af.flat(E_y), 1, args.f.shape[1], args.f.shape[2], args.f.shape[3]) #(i + 1/2, j + 1/2)
    E_z = af.tile(af.flat(E_z), 1, args.f.shape[1], args.f.shape[2], args.f.shape[3]) #(i + 1/2, j + 1/2)

    B_x = af.tile(af.flat(B_x), 1, args.f.shape[1], args.f.shape[2], args.f.shape[3]) #(i + 1/2, j + 1/2)
    B_y = af.tile(af.flat(B_y), 1, args.f.shape[1], args.f.shape[2], args.f.shape[3]) #(i + 1/2, j + 1/2)
    B_z = af.tile(af.flat(B_z), 1, args.f.shape[1], args.f.shape[2], args.f.shape[3]) #(i + 1/2, j + 1/2)
 
  else:
    E_x = af.tile(af.flat(E_x), 1, args.f.shape[1], args.f.shape[2], 1) #(i + 1/2, j + 1/2)
    E_y = af.tile(af.flat(E_y), 1, args.f.shape[1], args.f.shape[2], 1) #(i + 1/2, j + 1/2)
    E_z = af.tile(af.flat(E_z), 1, args.f.shape[1], args.f.shape[2], 1) #(i + 1/2, j + 1/2)

    B_x = af.tile(af.flat(B_x), 1, args.f.shape[1], args.f.shape[2], 1) #(i + 1/2, j + 1/2)
    B_y = af.tile(af.flat(B_y), 1, args.f.shape[1], args.f.shape[2], 1) #(i + 1/2, j + 1/2)
    B_z = af.tile(af.flat(B_z), 1, args.f.shape[1], args.f.shape[2], 1) #(i + 1/2, j + 1/2)

  F_x = charge_electron * (E_x + vel_y * B_z - vel_z * B_y) #(i + 1/2, j + 1/2)
  F_y = charge_electron * (E_y - vel_x * B_z + vel_z * B_x) #(i + 1/2, j + 1/2)
  F_z = charge_electron * (E_z - vel_y * B_x + vel_x * B_y) #(i + 1/2, j + 1/2)

  args.f      = f_interp_vel_3d(args, F_x, F_y, F_z, dt)

  # Convert to positionsExpanded:
  args.f = cks.convert.to_positionsExpanded(da, args.config, args.f)

  af.eval(args.f)
  return(args)

def time_integration(da, da_fields, args, time_array):

  data = np.zeros(time_array.size)

  glob  = da.createGlobalVec()
  local = da.createLocalVec()

  glob_field  = da_fields.createGlobalVec()
  local_field = da_fields.createLocalVec()

  # Convert to velocitiesExpanded:
  args.f = cks.convert.to_velocitiesExpanded(da, args.config, args.f)

  # Storing the value of density amplitude at t = 0
  # data[0] = af.max(cks.compute_moments.calculate_density(args))

  # Convert to positionsExpanded:
  args.f = cks.convert.to_positionsExpanded(da, args.config, args.f)
  args.f_half = args.f
  
  for time_index, t0 in enumerate(time_array[1:]):
    # Printing progress every 10 iterations
    # Printing only at rank = 0 to avoid multiple outputs:
    if(time_index%1 == 0 and da.getComm().rank == 0):
        print("Computing for Time =", t0)

    dt = time_array[1] - time_array[0]

    # Advection in position space:
    # args   = fields_step(da_fields, args, local_field, glob_field, dt)
    # args        = fields_step(da_fields, args, local_field, glob_field, dt, 1)
    # args.f_half = cks.communicate.communicate_distribution_function(da, args, local, glob)
    # Collision-Step:
    # args.f = collision_step_BGK(da, args, 0.5*dt)
    # args.f = cks.communicate.communicate_distribution_function(da, args, local, glob)
    # Advection in position space:
    # args   = fields_step(da_fields, args, local_field, glob_field, dt)
    # args.f = f_interp_2d(da, args, 0.25*dt)
    # args.f = cks.communicate.communicate_distribution_function(da, args, local, glob)
    # Fields Step(Advection in velocity space):
    args   = fields_step(da_fields, args, local_field, glob_field, dt)
    args.f = cks.communicate.communicate_distribution_function(da, args, local, glob)
    # Advection in position space:
    # args   = fields_step(da_fields, args, local_field, glob_field, dt)
    # args.f = f_interp_2d(da, args, 0.25*dt)
    # args.f = cks.communicate.communicate_distribution_function(da, args, local, glob)
    # Collision-Step:
    # args.f = collision_step_BGK(da, args, 0.5*dt)
    # args.f = cks.communicate.communicate_distribution_function(da, args, local, glob)
    # Advection in position space:
    # args   = fields_step(da_fields, args, local_field, glob_field, dt)
    # args.f = f_interp_2d(da, args, 0.25*dt)
    # args.f = cks.communicate.communicate_distribution_function(da, args, local, glob)
    
    # Convert to velocitiesExpanded:
    args.f = cks.convert.to_velocitiesExpanded(da, args.config, args.f)

    data[time_index + 1] = af.max(cks.compute_moments.calculate_density(args))

    # Convert to positionsExpanded:
    args.f = cks.convert.to_positionsExpanded(da, args.config, args.f)

  glob.destroy()
  local.destroy()
  glob_field.destroy()
  local_field.destroy()

  return(data, args.f)