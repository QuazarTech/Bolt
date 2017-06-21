# Importing parameter files which will be used in the run.
import params_files.N_32 as N_32
import params_files.N_64 as N_64
import params_files.N_128 as N_128
import params_files.N_256 as N_256
import params_files.N_512 as N_512

from mpi4py import MPI
import petsc4py
from petsc4py import PETSc

# Importing solver library functions
import setup_simulation
import cks.initialize as initialize
from cks.EM_fields_solver.electrostatic import solve_electrostatic_fields
import cks.evolve

import arrayfire as af
import numpy as np
import h5py

# config stores all the config objects for which the simulation is to be run:
config     = []
config_32  = setup_simulation.configuration_object(N_32)
config.append(config_32)
config_64  = setup_simulation.configuration_object(N_64)
config.append(config_64)
config_128 = setup_simulation.configuration_object(N_128)
config.append(config_128)
config_256 = setup_simulation.configuration_object(N_256)
config.append(config_256)
config_512 = setup_simulation.configuration_object(N_512)
config.append(config_512)

petsc4py.init()

# Declaring the communicator:
comm = PETSc.COMM_WORLD.tompi4py()

# We assume that all the parameter files also have
# the same number of num_devices mentioned.
af.set_device(comm.rank%N_32.num_devices)

global_time = np.zeros(1)

print("Device info for rank", comm.rank)
af.info()

for i in range(len(config)):
  time_start = MPI.Wtime() # Starting the timer
  time_array = setup_simulation.time_array(config[i])

  # Getting the resolutions of position and velocity space:
  N_y     = config[i].N_y
  N_x     = config[i].N_x
  N_vel_y = config[i].N_vel_y
  N_vel_x = config[i].N_vel_x
  N_vel_z = config[i].N_vel_z
  N_ghost = config[i].N_ghost

  if(comm.rank == 0):
    print() # Insert blank line
    print("Running CKS for N =", N_x)
    print() # Insert blank line

  # Declaring distributed array object which automates the domain decomposition:
  # Additionally, it is also used to take care of the boundary conditions:
  da = PETSc.DMDA().create([N_y, N_x],\
                           dof = (N_vel_y * N_vel_x * N_vel_z),\
                           stencil_width = N_ghost,\
                           boundary_type = ('periodic', 'periodic'),\
                           proc_sizes = (PETSc.DECIDE, PETSc.DECIDE), \
                           stencil_type = 1, \
                           comm = comm
                          ) 

  da_fields = PETSc.DMDA().create([N_y, N_x],\
                                  dof = 1,\
                                  stencil_width = N_ghost,\
                                  boundary_type = da.getBoundaryType(),\
                                  proc_sizes = da.getProcSizes(), \
                                  stencil_type = 1, \
                                  comm = da.getComm()
                                 )

  # Declaring global vectors to export the final distribution function:
  global_vector    = da.createGlobalVec()
  global_vec_value = da.getVecArray(global_vector)

  # Changing name of object so that dataset may be read from h5py
  PETSc.Object.setName(global_vector, 'distribution_function')
  
  viewer = PETSc.Viewer().createHDF5('distribution_function_data_files/ck/ck_distribution_function_' \
                                      + str(N_x) + '.h5', 'w', comm = comm
                                    )

  # Obtaining the velocity and position arrays:
  x_center = initialize.calculate_x_center(da, config[i])
  y_center = initialize.calculate_y_center(da, config[i])

  x_left   = initialize.calculate_x_left(da, config[i])
  y_bottom = initialize.calculate_y_bottom(da, config[i])

  x_right = initialize.calculate_x_right(da, config[i])
  y_top   = initialize.calculate_y_top(da, config[i])

  vel_x, vel_y, vel_z = initialize.calculate_velocities(da, config[i]) #velocitiesExpanded form

  # Initializing the value for distribution function:
  f_initial = initialize.f_initial(da, config[i])

  # We define an object args that holds, the position arrays, 
  # velocity arrays, distribution function and field quantities.
  # By this manner, if we have the args object for any time-step, all
  # the information about the system may be retrieved:
  class args:
    def __init__(self):
      pass

  args.config = config[i]
  args.f      = f_initial

  args.vel_x = vel_x
  args.vel_y = vel_y
  args.vel_z = vel_z

  args.x_center = x_center
  args.y_center = y_center

  pert_real = config[i].pert_real
  pert_imag = config[i].pert_imag

  k_x = config[i].k_x
  k_y = config[i].k_y

  charge_electron = config[i].charge_electron
  
  args.f    = cks.convert.to_velocitiesExpanded(da, config[i], args.f)
  rho_array = config[i].charge_electron * (cks.compute_moments.calculate_density(args) - \
                                           config[i].rho_background
                                          )
 
  # Obtaining the left-bottom corner coordinates 
  # of the left-bottom corner cell in the local zone considered:
  # We also obtain the size of the local zone:
  ((j_bottom, i_left), (N_y_local, N_x_local)) = da_fields.getCorners()

  rho_array = af.moddims(rho_array,\
                         N_y_local + 2 * N_ghost,\
                         N_x_local + 2 * N_ghost
                        )

  rho_array = np.array(rho_array)[N_ghost:-N_ghost,\
                                  N_ghost:-N_ghost
                                 ]
  # This function returns the values of fields at (i + 0.5, j + 0.5)
  args.E_x, args.E_y =\
  solve_electrostatic_fields(da_fields, config[i], rho_array)

  # Interpolating to obtain the values at the Yee-Grid
  args.E_x = 0.5 * (args.E_x + af.shift(args.E_x, 1, 0))
  args.E_y = 0.5 * (args.E_y + af.shift(args.E_y, 0, 1))

  # We define da_fields with dof = 6 to allow application of boundary conditions
  # for all the fields quantities in a single step.
  if(config[i].fields_solver == 'fdtd'):
    da_fields.destroy()
    da_fields = PETSc.DMDA().create([N_y, N_x],\
                                    dof = 6,\
                                    stencil_width = N_ghost,\
                                    boundary_type = da.getBoundaryType(),\
                                    proc_sizes = da.getProcSizes(), \
                                    stencil_type = 1, \
                                    comm = da.getComm()
                                   )

  args.f = cks.convert.to_positionsExpanded(da, args.config, args.f)

  # The following quantities are defined on the Yee-Grid:
  args.E_z = af.constant(0, x_left.shape[0], y_bottom.shape[1], dtype=af.Dtype.f64)
  args.B_x = af.constant(0, x_left.shape[0], y_center.shape[1], dtype=af.Dtype.f64)
  args.B_y = af.constant(0, x_center.shape[0], y_bottom.shape[1], dtype=af.Dtype.f64)
  args.B_z = af.constant(0, x_center.shape[0], x_center.shape[1], dtype=af.Dtype.f64)

  global_data   = np.zeros(time_array.size) 
  data, f_final = cks.evolve.time_integration(da, da_fields, args, time_array)

  # Passing the values non-inclusive of ghost cells:
  global_vec_value[:] = np.array(f_final[N_ghost:-N_ghost, N_ghost:-N_ghost, :])
  viewer(global_vector)

  comm.Reduce(data,\
              global_data,\
              op = MPI.MAX,\
              root = 0
             )

  if(comm.rank == 0):
    h5f = h5py.File('density_data_files/ck/ck_density_data_' + str(N_x) + '.h5', 'w')
    h5f.create_dataset('density_data', data = global_data)
    h5f.create_dataset('time', data = time_array)
    h5f.close()

  time_end     = MPI.Wtime() # Ending the timer
  time_elapsed = np.array([time_end - time_start])
  
  comm.Reduce(time_elapsed, global_time, op = MPI.MAX, root = 0)
  
  if(comm.rank == 0):
    print("Time Elapsed =", global_time[0])