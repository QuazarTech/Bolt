from mpi4py import MPI
import petsc4py
from petsc4py import PETSc

# Importing solver library functions
import setup_simulation
import cks.initialize as initialize
import cks.evolve as evolve

import importlib
pylab_found = importlib.util.find_spec("pylab")
import arrayfire as af
import numpy as np

import params

if(pylab_found != None):
  import pylab as pl

  # Setting plot parameters:
  pl.rcParams['figure.figsize']  = 12, 7.5
  pl.rcParams['figure.dpi']      = 300
  pl.rcParams['image.cmap']      = 'jet'
  pl.rcParams['lines.linewidth'] = 1.5
  pl.rcParams['font.family']     = 'serif'
  pl.rcParams['font.weight']     = 'bold'
  pl.rcParams['font.size']       = 20  
  pl.rcParams['font.sans-serif'] = 'serif'
  pl.rcParams['text.usetex']     = True
  pl.rcParams['axes.linewidth']  = 1.5
  pl.rcParams['axes.titlesize']  = 'medium'
  pl.rcParams['axes.labelsize']  = 'medium'

  pl.rcParams['xtick.major.size'] = 8     
  pl.rcParams['xtick.minor.size'] = 4     
  pl.rcParams['xtick.major.pad']  = 8     
  pl.rcParams['xtick.minor.pad']  = 8     
  pl.rcParams['xtick.color']      = 'k'     
  pl.rcParams['xtick.labelsize']  = 'medium'
  pl.rcParams['xtick.direction']  = 'in'    

  pl.rcParams['ytick.major.size'] = 8     
  pl.rcParams['ytick.minor.size'] = 4     
  pl.rcParams['ytick.major.pad']  = 8     
  pl.rcParams['ytick.minor.pad']  = 8     
  pl.rcParams['ytick.color']      = 'k'     
  pl.rcParams['ytick.labelsize']  = 'medium'
  pl.rcParams['ytick.direction']  = 'in' 

# Setting up the configuration object along with the time array
config      = setup_simulation.configuration_object(params)
time_array  = setup_simulation.time_array(config)
num_devices = params.num_devices

# Getting the resolutions of position and velocity space:
N_y     = config.N_y
N_x     = config.N_x
N_vel_y = config.N_vel_y
N_vel_x = config.N_vel_x
N_ghost = config.N_ghost

petsc4py.init()

# Declaring the communicator:
comm = PETSc.COMM_WORLD.tompi4py()
af.set_device(comm.rank%num_devices)

# Declaring distributed array object which automates the domain decomposition:
# Additionally, it is also used to take care of the boundary conditions:
da = PETSc.DMDA().create([N_y, N_x],\
                         dof = (N_vel_y * N_vel_x),\
                         stencil_width = N_ghost,\
                         boundary_type = ('periodic', 'periodic'),\
                         proc_sizes = (PETSc.DECIDE, PETSc.DECIDE), \
                         stencil_type = 1, \
                         comm = comm
                        ) 

# Obtaining the left-bottom corner coordinates 
# of the left-bottom corner cell in the local zone considered:
((j_bottom_left, i_bottom_left), (N_y_local, N_x_local)) = da.getCorners()

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
x_left   = initialize.calculate_x_left(da, config)
vel_x    = initialize.calculate_vel_x(da, config)
y_center = initialize.calculate_y_center(da, config)
y_bottom = initialize.calculate_y_bottom(da, config)
vel_y    = initialize.calculate_vel_y(da, config)

# Initializing the value for distribution function:
f_initial = initialize.f_initial(da, config)

# We define an object args that holds, the position arrays, 
# velocity arrays, distribution function and field quantities.
# By this manner, if we have the args object for any time-step, all
# the information about the system may be retrieved:
class args:
  def __init__(self):
    pass

args.config   = config
args.f        = f_initial
args.vel_x    = vel_x
args.vel_y    = vel_y
args.x_center = x_center
args.y_center = y_center

pert_real = config.pert_real
pert_imag = config.pert_imag
k_x       = config.k_x
k_y       = config.k_y

charge_electron = config.charge_electron

# The following quantities are defined on the Yee-Grid:
args.E_x = charge_electron * k_x/(k_x**2 + k_y**2) *\
           (pert_real * af.sin(k_x*x_center[:, :, 0, 0] + k_y*y_bottom[:, :, 0, 0]) +\
            pert_imag * af.cos(k_x*x_center[:, :, 0, 0] + k_y*y_bottom[:, :, 0, 0])
           ) #(i + 1/2, j)

args.E_y = charge_electron * k_y/(k_x**2 + k_y**2) *\
           (pert_real * af.sin(k_x*x_left[:, :, 0, 0] + k_y*y_center[:, :, 0, 0]) +\
            pert_imag * af.cos(k_x*x_left[:, :, 0, 0] + k_y*y_center[:, :, 0, 0])
           ) #(i, j + 1/2)

args.B_z = af.constant(0, x_center.shape[0], x_center.shape[1], dtype=af.Dtype.f64) #(i + 1/2, j + 1/2)

args.B_x = af.constant(0, x_center.shape[0], x_center.shape[1], dtype=af.Dtype.f64) #(i, j + 1/2)
args.B_y = af.constant(0, x_center.shape[0], x_center.shape[1], dtype=af.Dtype.f64) #(i + 1/2, j)
args.E_z = af.constant(0, x_center.shape[0], x_center.shape[1], dtype=af.Dtype.f64) #(i, j)

# Global data holds the information of the density amplitude for the entire physical domain:
global_data   = np.zeros(time_array.size) 
data, f_final = evolve.time_integration(da, args, time_array)

vel_x_max = config.vel_x_max
vel_y_max = config.vel_y_max
dv_x      = (2*vel_x_max)/(N_vel_x - 1)
dv_y      = (2*vel_y_max)/(N_vel_y - 1)

# Passing the values non-inclusive of ghost cells:
f_final             = f_final[N_ghost:-N_ghost, N_ghost:-N_ghost, :, :]
global_vec_value[:] = np.array(af.moddims(f_final, N_y_local, N_x_local, N_vel_x * N_vel_y))

viewer(global_vector)

# Performing a reduce operation to obtain the global data:
comm.Reduce(data,\
            global_data,\
            op = MPI.MAX,\
            root = 0
           )

# Plotting/Export of the global-data:
if(comm.rank == 0):
  if(pylab_found != None):
    pl.plot(time_array, global_data)
    pl.xlabel('Time')
    pl.ylabel(r'$MAX(\delta \rho(x))$')
    pl.savefig('plot.png')
  else:
    import h5py
    h5f = h5py.File('density_data_ck.h5', 'w')
    h5f.create_dataset('density_amplitude', data = global_data)
    h5f.create_dataset('time', data = time_array)
    h5f.close()
