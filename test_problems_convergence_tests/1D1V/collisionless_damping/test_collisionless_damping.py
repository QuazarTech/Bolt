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
import cks.initialize
import cks.evolve
import lts.initialize
import lts.evolve
import lts.export

import arrayfire as af
import numpy as np
import h5py

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

def run_ck():
  
  for i in range(len(config)):
    time_array = setup_simulation.time_array(config[i])

    # Getting the resolutions of position and velocity space:
    N_y     = config[i].N_y
    N_x     = config[i].N_x
    N_vel_y = config[i].N_vel_y
    N_vel_x = config[i].N_vel_x
    N_ghost = config[i].N_ghost

    print("Running CK for N =", N_x)
    petsc4py.init()

    # Declaring the communicator:
    comm = PETSc.COMM_WORLD.tompi4py()

    # Declaring distributed array object which automates the domain decomposition:
    da = PETSc.DMDA().create([N_y, N_x],\
                             dof = (N_vel_y * N_vel_x),\
                             stencil_width = N_ghost,\
                             boundary_type = ('periodic', 'periodic'),\
                             proc_sizes = (PETSc.DECIDE, PETSc.DECIDE), \
                             stencil_type = 1, \
                             comm = comm
                            ) 

    ((j_bottom_left, i_bottom_left), (N_y_local, N_x_local)) = da.getCorners()

    # Declaring global vectors to export the final distribution function:
    global_vector    = da.createGlobalVec()
    global_vec_value = da.getVecArray(global_vector)

    # Changing name of object so that dataset may be read from h5py
    PETSc.Object.setName(global_vector, 'distribution_function')
    viewer = PETSc.Viewer().createHDF5('ck_distribution_function_' + str(N_x) + '.h5', 'w', comm = comm)

    # Printing only when rank = 0 to avoid multiple outputs:
    if(comm.rank == 0):
      print(af.info())

    x     = cks.initialize.calculate_x(da, config[i])
    vel_x = cks.initialize.calculate_vel_x(da, config[i])
    y     = cks.initialize.calculate_y(da, config[i])
    vel_y = cks.initialize.calculate_vel_y(da, config[i])

    f_initial = cks.initialize.f_initial(da, config[i])

    class args:
      def __init__(self):
        pass

    args.config = config[i]
    args.f      = f_initial
    args.vel_x  = vel_x
    args.vel_y  = vel_y
    args.x      = x
    args.y      = y

    pert_real = config[i].pert_real
    pert_imag = config[i].pert_imag
    k_x       = config[i].k_x
    k_y       = config[i].k_y

    charge_electron = config[i].charge_electron

    args.E_x = charge_electron * k_x/(k_x**2 + k_y**2) *\
               (pert_real * af.sin(k_x*x[:, :, 0, 0] + k_y*y[:, :, 0, 0]) +\
                pert_imag * af.cos(k_x*x[:, :, 0, 0] + k_y*y[:, :, 0, 0])
               )

    args.E_y = charge_electron * k_y/(k_x**2 + k_y**2) *\
               (pert_real * af.sin(k_x*x[:, :, 0, 0] + k_y*y[:, :, 0, 0]) +\
                pert_imag * af.cos(k_x*x[:, :, 0, 0] + k_y*y[:, :, 0, 0])
               )

    args.B_z = af.constant(0, x.shape[0], x.shape[1], dtype=af.Dtype.f64)
    args.B_x = af.constant(0, x.shape[0], x.shape[1], dtype=af.Dtype.f64)
    args.B_y = af.constant(0, x.shape[0], x.shape[1], dtype=af.Dtype.f64)
    args.E_z = af.constant(0, x.shape[0], x.shape[1], dtype=af.Dtype.f64)

    global_data   = np.zeros(time_array.size) 
    data, f_final = cks.evolve.time_integration(da, args, time_array)

    vel_x_max = config[i].vel_x_max
    vel_y_max = config[i].vel_y_max
    dv_x      = (2*vel_x_max)/(N_vel_x - 1)
    dv_y      = (2*vel_y_max)/(N_vel_y - 1)

    f_final       = f_final[N_ghost:-N_ghost, N_ghost:-N_ghost, :, :]
    normalization = af.sum(cks.initialize.f_background(da, config[i])) * dv_x * dv_y/(x.shape[0] * x.shape[1])
    f_background  = (cks.initialize.f_background(da, config[i])/normalization)[N_ghost:-N_ghost,\
                                                                            N_ghost:-N_ghost, :, :
                                                                           ]
    f_perturbed   = f_final - f_background

    global_vec_value[:] = np.array(af.moddims(f_perturbed, N_y_local, N_x_local, N_vel_x * N_vel_y))
    viewer(global_vector)

    comm.Reduce(data,\
                global_data,\
                op = MPI.MAX,\
                root = 0
               )

    if(comm.rank == 0):
      h5f = h5py.File('ck_density_data_' + str(N_x) + '.h5', 'w')
      h5f.create_dataset('density_data', data = global_data)
      h5f.close()

def run_lt():
  
  for i in range(len(config)):
    print("Running LT for N =", config[i].N_x)
    time_array = setup_simulation.time_array(config[i])
    delta_f_hat_initial = lts.initialize.init_delta_f_hat(config[i])
    delta_rho_hat, delta_f_hat_final = lts.evolve.time_integration(config[i], delta_f_hat_initial, time_array)
    
    lts.export.export_4D_distribution_function(config[i], delta_f_hat_final)
    h5f = h5py.File('lt_density_data_' + str(config[i].N_x) + '.h5', 'w')
    h5f.create_dataset('density_data', data = delta_rho_hat)
    h5f.close()

def test_case():
  run_ck()
  run_lt()

  # error = np.zeros(len(config))

  # for i in range(len(config))

  #   error[i] = af.sum(af.abs(af.to_array(f_dist) + f_background[3:-3] - f_final[3:-3]))/f_dist.size

  # x = np.log10(2**np.arange(5, 10))

  # poly = np.polyfit(x, np.log10(error), 1)

  # assert(abs(poly[0]+2)<0.2)

# run_ck()
run_lt()