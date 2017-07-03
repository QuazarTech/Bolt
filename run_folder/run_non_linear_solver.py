# Importing dependencies:
from mpi4py import MPI
import petsc4py, sys
from petsc4py import PETSc

import arrayfire as af
import numpy as np
import pylab as pl
import params


# Importing solver library functions
import setup_simulation
import non_linear_solver.initialize as initialize
import non_linear_solver.evolve as evolve
import non_linear_solver.compute_moments
import non_linear_solver.communicate
from non_linear_solver.EM_fields_solver.electrostatic import solve_electrostatic_fields, fft_poisson

# Setting up the configuration object along with the time array.
config      = setup_simulation.configuration_object(params)
time_array  = setup_simulation.time_array(config)
num_devices = params.num_devices

# Getting the resolutions of position and velocity space:
N_y     = config.N_y
N_x     = config.N_x

N_vel_y = config.N_vel_y
N_vel_x = config.N_vel_x
N_vel_z = config.N_vel_z

N_ghost = config.N_ghost

petsc4py.init(sys.argv)

# Declaring the communicator:
comm = PETSc.COMM_WORLD.tompi4py()
af.set_device(comm.rank%num_devices)

if(config.bc_in_x == 'dirichlet'):
  bc_in_x = 'ghosted'

else:
  bc_in_x = 'periodic'

if(config.bc_in_y == 'dirichlet'):
  bc_in_y = 'ghosted'

else:
  bc_in_y = 'periodic'

# Declaring distributed array object which automates the domain decomposition:
# Additionally, it is also used to take care of the boundary conditions:
da = PETSc.DMDA().create([N_y, N_x],\
                         dof = (N_vel_y * N_vel_x * N_vel_z),\
                         stencil_width = N_ghost,\
                         boundary_type = (bc_in_x, bc_in_y),\
                         proc_sizes = (PETSc.DECIDE, PETSc.DECIDE), \
                         stencil_type = 1, \
                         comm = comm
                        ) 

da_fields = PETSc.DMDA().create([N_y, N_x],\
                                dof = 6,\
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
viewer = PETSc.Viewer().createHDF5('ck_distribution_function.h5', 'w', comm = comm)

# Printing only when rank = 0 to avoid multiple outputs:
if(comm.rank == 0):
  print(af.info())

# Obtaining the velocity and position arrays:
x_center = initialize.calculate_x_center(da, config)
y_center = initialize.calculate_y_center(da, config)

x_left   = initialize.calculate_x_left(da, config)
y_bottom = initialize.calculate_y_bottom(da, config)

x_right = initialize.calculate_x_right(da, config)
y_top   = initialize.calculate_y_top(da, config)

vel_x, vel_y, vel_z = initialize.calculate_velocities(da, config) #velocitiesExpanded form

# We define an object args that holds, the position arrays, 
# velocity arrays, distribution function and field quantities.
# By this manner, if we have the args object for any time-step, all
# the information about the system may be retrieved:
class args:
  def __init__(self):
    pass

args.config = config

args.vel_x = vel_x
args.vel_y = vel_y
args.vel_z = vel_z

args.x_center = x_center
args.y_center = y_center

pert_real = config.pert_real
pert_imag = config.pert_imag

k_x = config.k_x
k_y = config.k_y

# Initializing the value for distribution function:
args = initialize.log_f_initial(da, args)

charge_electron = config.charge_electron

# Assigning the initial values for the electric fields:
args.log_f = non_linear_solver.convert.to_velocitiesExpanded(da, config, args.log_f)
rho_array  = config.charge_electron * (non_linear_solver.compute_moments.calculate_density(args) - \
                                      config.rho_background
                                     )

# Obtaining the left-bottom corner coordinates 
# of the left-bottom corner cell in the local zone considered:
# We also obtain the size of the local zone:
((j_bottom, i_left), (N_y_local, N_x_local)) = da_fields.getCorners()

rho_array = af.moddims(rho_array,\
                       N_y_local + 2 * N_ghost,\
                       N_x_local + 2 * N_ghost
                      )

rho_array = (rho_array)[N_ghost:-N_ghost,\
                                N_ghost:-N_ghost
                               ]

args.E_x = af.constant(0, x_left.shape[0], y_center.shape[1], dtype=af.Dtype.f64)
args.E_y = af.constant(0, x_center.shape[0], y_bottom.shape[1], dtype=af.Dtype.f64)
args.E_z = af.constant(0, x_left.shape[0], y_bottom.shape[1], dtype=af.Dtype.f64)
args.B_x = af.constant(0, x_left.shape[0], y_center.shape[1], dtype=af.Dtype.f64)
args.B_y = af.constant(0, x_center.shape[0], y_bottom.shape[1], dtype=af.Dtype.f64)
args.B_z = af.constant(0, x_center.shape[0], x_center.shape[1], dtype=af.Dtype.f64)

# This function returns the values of fields at (i + 0.5, j + 0.5)
args.E_x[3:-3, 3:-3], args.E_y[3:-3, 3:-3] = fft_poisson(rho_array, config.dx, config.dy)

# pl.contourf(np.array(args.E_x), 100)
# pl.colorbar()
# pl.show()
# pl.clf()
# pl.contourf(np.array(args.E_y), 100)
# pl.colorbar()
# pl.show()

# glob  = da_fields.createGlobalVec()
# local = da_fields.createLocalVec()

# args = non_linear_solver.communicate.communicate_fields(da_fields, args, local, glob)

# solve_electrostatic_fields(da_fields, config, rho_array)

# Interpolating to obtain the values at the Yee-Grid
args.E_x = 0.5 * (args.E_x + af.shift(args.E_x, 1, 0)) #(i+0.5, j)
args.E_y = 0.5 * (args.E_y + af.shift(args.E_y, 0, 1)) #(i, j+0.5)

# args.E_x = charge_electron * k_x/(k_x**2 + k_y**2) *\
#            (pert_real * af.sin(k_x*x_center[:, :, 0, 0] + k_y*y_bottom[:, :, 0, 0]) +\
#             pert_imag * af.cos(k_x*x_center[:, :, 0, 0] + k_y*y_bottom[:, :, 0, 0])
#            ) #(i + 1/2, j)

# args.E_y = charge_electron * k_y/(k_x**2 + k_y**2) *\
#            (pert_real * af.sin(k_x*x_left[:, :, 0, 0] + k_y*y_center[:, :, 0, 0]) +\
#             pert_imag * af.cos(k_x*x_left[:, :, 0, 0] + k_y*y_center[:, :, 0, 0])
#            ) #(i, j + 1/2)
           
# We define da_fields with dof = 6 to allow application of boundary conditions
# for all the fields quantities in a single step.
if(config.fields_solver == 'fdtd'):
  da_fields.destroy()
  da_fields = PETSc.DMDA().create([N_y, N_x],\
                                  dof = 6,\
                                  stencil_width = N_ghost,\
                                  boundary_type = da.getBoundaryType(),\
                                  proc_sizes = da.getProcSizes(), \
                                  stencil_type = 1, \
                                  comm = da.getComm()
                                 )

args.log_f = non_linear_solver.convert.to_positionsExpanded(da, args.config, args.log_f)

# The following quantities are defined on the Yee-Grid:


# Global data holds the information of the density amplitude for the entire physical domain:
global_data   = np.zeros(time_array.size) 
data, f_final = evolve.time_integration(da, da_fields, args, time_array)

# Passing the values non-inclusive of ghost cells:
global_vec_value[:] = np.array(f_final[N_ghost:-N_ghost, N_ghost:-N_ghost, :])
viewer(global_vector)

# Performing a reduce operation to obtain the global data:
comm.Reduce(data,\
            global_data,\
            op = MPI.MAX,\
            root = 0
           )

# Export of the global-data:
if(comm.rank == 0):
  import h5py
  h5f = h5py.File('ck_density_data.h5', 'w')
  h5f.create_dataset('density_amplitude', data = global_data - config.rho_background)
  h5f.create_dataset('time', data = time_array)
  h5f.close()