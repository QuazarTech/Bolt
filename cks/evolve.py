import numpy as np
import arrayfire as af
import cks.initialize as initialize

from cks.compute_moments import calculate_density, calculate_vel_bulk_x,\
                                calculate_vel_bulk_y, calculate_mom_bulk_x,\
                                calculate_mom_bulk_y, calculate_temperature

from petsc4py import PETSc

def communicate_distribution_function(da, args, local, glob):

  # Accessing the values of the global and local Vectors
  local_value = da.getVecArray(local)
  glob_value  = da.getVecArray(glob)

  N_ghost = args.config.N_ghost
  N_vel_x = args.config.N_vel_x
  N_vel_y = args.config.N_vel_y

  # Obtaining the left corner coordinates for the local zone considered:
  ((j_bottom_left, i_bottom_left), (N_y_local, N_x_local)) = da.getCorners()

  # Changing the dimensions so that the values in the 4D af.Array
  # can be stored in the 3D PETSc.Vec:
  local_value[:] = np.array(af.moddims(args.f,\
                                       N_y_local + 2*N_ghost, \
                                       N_x_local + 2*N_ghost, \
                                       N_vel_x * N_vel_y, \
                                       1
                                       )
                            )
  
  # Global value is non-inclusive of the ghost-zones:
  glob_value[:] = (local_value[:])[N_ghost:-N_ghost,\
                                   N_ghost:-N_ghost,\
                                   :
                                  ]

  # The following function takes care of the boundary conditions, and interzonal communications:
  da.globalToLocal(glob, local)

  # Converting back from PETSc.Vec to af.Array:
  f_updated = af.moddims(af.to_array(local_value[:]),\
                         N_y_local + 2*N_ghost, \
                         N_x_local + 2*N_ghost, \
                         N_vel_y, \
                         N_vel_x
                        )

  return(f_updated)

def communicate_fields(da, config, local_field, local, glob):

  # Accessing the values of the global and local Vectors
  local_value = da.getVecArray(local)
  glob_value  = da.getVecArray(glob)

  N_ghost = config.N_ghost

  # Obtaining the left corner coordinates for the local zone considered:
  ((j_bottom_left, i_bottom_left), (N_y_local, N_x_local)) = da.getCorners()

  local_value[:] = np.array(local_field)
  
  glob_value[:] = (local_value[:])[N_ghost:-N_ghost,\
                                   N_ghost:-N_ghost
                                  ]

  da.globalToLocal(glob, local)

  field_updated = af.to_array(local_value[:])
  return(field_updated)

def f_MB(da, args):

  config = args.config
  f      = args.f
  vel_x  = args.vel_x
  vel_y  = args.vel_y

  dv_x = af.sum(vel_x[0, 0, 0, 1]-vel_x[0, 0, 0, 0])
  dv_y = af.sum(vel_y[0, 0, 1, 0]-vel_y[0, 0, 0, 0])
  
  mass_particle      = config.mass_particle
  boltzmann_constant = config.boltzmann_constant

  if(config.mode == '2V'):
    n          = af.tile(calculate_density(args), 1, 1, f.shape[2], f.shape[3])
    T          = af.tile(calculate_temperature(args), 1, 1, f.shape[2], f.shape[3])
    vel_bulk_x = af.tile(calculate_vel_bulk_x(args), 1, 1, f.shape[2], f.shape[3])
    vel_bulk_y = af.tile(calculate_vel_bulk_y(args), 1, 1, f.shape[2], f.shape[3])
    
    f_MB = n * (mass_particle/(2*np.pi*boltzmann_constant*T)) * \
           af.exp(-mass_particle*(vel_x - vel_bulk_x)**2/(2*boltzmann_constant*T)) * \
           af.exp(-mass_particle*(vel_y - vel_bulk_y)**2/(2*boltzmann_constant*T))  
  
  else:
    n          = af.tile(calculate_density(args), 1, 1, f.shape[2], f.shape[3])
    T          = af.tile(calculate_temperature(args), 1, 1, f.shape[2], f.shape[3])
    vel_bulk_x = af.tile(calculate_vel_bulk_x(args), 1, 1, f.shape[2], f.shape[3])
    
    f_MB = n*af.sqrt(mass_particle/(2*np.pi*boltzmann_constant*T))*\
             af.exp(-mass_particle*(vel_x-vel_bulk_x)**2/(2*boltzmann_constant*T))


  normalization = af.sum(initialize.f_background(da, config))*dv_x*dv_y/(vel_x.shape[0] * vel_x.shape[1])
  f_MB          = f_MB/normalization

  af.eval(f_MB)
  return(f_MB)

def collision_step(da, args, dt):

  tau = args.config.tau
  f   = args.f 

  f0             = f_MB(da, args)
  f_intermediate = f - (dt/2)*(f - f0)/tau
  f_final        = f - (dt)  *(f_intermediate - f0)/tau

  af.eval(f_final)
  return(f_final)

def fields_step(da, args, dt):
  
  config = args.config
  f      = args.f
  vel_x  = args.vel_x
  vel_y  = args.vel_y

  charge_electron = config.charge_electron

  from cks.interpolation_routines import f_interp_vel_2d
  from cks.fdtd import fdtd, fdtd_grid_to_ck_grid

  glob  = da.createGlobalVec()
  local = da.createLocalVec()

  E_x = args.E_x
  E_y = args.E_y
  E_z = args.E_z

  B_x = args.B_x
  B_y = args.B_y
  B_z = args.B_z

  J_x = charge_electron * calculate_mom_bulk_x(args)
  J_y = charge_electron * calculate_mom_bulk_y(args) 
  J_z = af.constant(0, J_x.shape[0], J_x.shape[1])

  J_x = 0.5 * (J_x + af.shift(J_x, 0, -1))
  J_y = 0.5 * (J_y + af.shift(J_y, -1, 0))

  J_x = communicate_fields(da, config, J_x, local, glob)
  J_y = communicate_fields(da, config, J_y, local, glob)

  E_x, E_y, E_z, B_x_new, B_y_new, B_z_new = fdtd(da, config,\
                                                  E_x, E_y, E_z,\
                                                  B_x, B_y, B_z,\
                                                  J_x, J_y, J_z,\
                                                  dt
                                                 )

  args.B_x = B_x_new
  args.B_y = B_y_new
  args.B_z = B_z_new
  args.E_x = E_x
  args.E_y = E_y
  args.E_z = E_z

  # To account for half-time steps:
  B_x = 0.5 * (B_x + B_x_new)
  B_y = 0.5 * (B_y + B_y_new)
  B_z = 0.5 * (B_z + B_z_new)

  E_x, E_y, E_z, B_x, B_y, B_z = fdtd_grid_to_ck_grid(da, config, E_x, E_y, E_z, B_x, B_y, B_z)

  E_x = af.tile(E_x, 1, 1, f.shape[2], f.shape[3])
  E_y = af.tile(E_y, 1, 1, f.shape[2], f.shape[3])
  B_z = af.tile(B_z, 1, 1, f.shape[2], f.shape[3])

  F_x = charge_electron * (E_x + vel_y * B_z)
  F_y = charge_electron * (E_y - vel_x * B_z)

  args.f = f_interp_vel_2d(args, F_x, F_y, dt)
    
  af.eval(args.f)
  return(args)

def time_integration(da, args, time_array):
    
  data = np.zeros(time_array.size)

  glob  = da.createGlobalVec()
  local = da.createLocalVec()

  da_fields = PETSc.DMDA().create([args.config.N_y, args.config.N_x],\
                                  stencil_width = args.config.N_ghost,\
                                  boundary_type = ('periodic', 'periodic'),\
                                  proc_sizes = (PETSc.DECIDE, PETSc.DECIDE), \
                                  stencil_type = 1, \
                                  comm = da.getComm()
                                  ) 

  from cks.interpolation_routines import f_interp_2d
  
  for time_index, t0 in enumerate(time_array):
    if(time_index%10 == 0 and da.getComm().rank == 0):
        print("Computing for Time = ", t0)

    dt = time_array[1] - time_array[0]

    args.f = f_interp_2d(da, args, 0.25*dt)
    args.f = communicate_distribution_function(da, args, local, glob)
    args.f = collision_step(da, args, 0.5*dt)
    args.f = communicate_distribution_function(da, args, local, glob)
    args.f = f_interp_2d(da, args, 0.25*dt)
    args.f = communicate_distribution_function(da, args, local, glob)
    args   = fields_step(da_fields, args, dt)
    args.f = communicate_distribution_function(da, args, local, glob)
    args.f = f_interp_2d(da, args, 0.25*dt)
    args.f = communicate_distribution_function(da, args, local, glob)
    args.f = collision_step(da, args, 0.5*dt)
    args.f = communicate_distribution_function(da, args, local, glob)
    args.f = f_interp_2d(da, args, 0.25*dt)
    args.f = communicate_distribution_function(da, args, local, glob)
      
    data[time_index] = af.max(calculate_density(args))

  return(data, args.f)


