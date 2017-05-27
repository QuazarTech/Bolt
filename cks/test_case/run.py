from mpi4py import MPI
import petsc4py
from petsc4py import PETSc 

import cks.initialize as initialize
import cks.evolve as evolve

import matplotlib as mpl 
mpl.use('Agg')
import pylab as pl

import arrayfire as af
import numpy as np

import params

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

config     = set.configuration_object(params)
time_array = set.time_array(config)

# Getting the resolutions of position and velocity space:
N_y     = config.N_y
N_x     = config.N_x
N_vel_y = config.N_vel_y
N_vel_x = config.N_vel_x
N_ghost = config.N_ghost

petsc4py.init()

comm = PETSc.COMM_WORLD.tompi4py()

da = PETSc.DMDA().create([N_y, N_x],\
                         dof = (N_vel_y * N_vel_x),\
                         stencil_width = N_ghost,\
                         boundary_type = ('periodic', 'periodic'),\
                         proc_sizes = (PETSc.DECIDE, PETSc.DECIDE), \
                         stencil_type = 1, \
                         comm = comm
                        ) 

global_vector    = da.createGlobalVec()
global_vec_value = da.getVecArray()

PETSc.Object.setName(global_vector, 'distribution_function')
viewer = PETSc.Viewer().createHDF5('ck_distribution_funtion.h5', 'w', comm = comm)

if(comm.rank == 0):
  print(af.info())

x     = initialize.calculate_x(da, config)
vel_x = initialize.calculate_vel_x(da, config)
y     = initialize.calculate_y(da, config)
vel_y = initialize.calculate_vel_y(da, config)

f_initial = initialize.f_initial(da, config)

class args:
    pass

args.config = config
args.f      = f_initial
args.vel_x  = vel_x
args.vel_y  = vel_y
args.x      = x
args.y      = y

pert_real = config.pert_real
pert_imag = config.pert_imag
k_x       = config.k_x
k_y       = config.k_y

charge_electron = config.charge_electron

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
data, f_final = evolve.time_integration(da, args, time_array)

global_vec_value = np.array(af.moddims(f_final, N_x, N_y, N_vel_x * N_vel_y))
viewer(global_vector)

comm.Reduce(data,\
            global_data,\
            op = MPI.MAX,\
            root = 0
           )

if(comm.rank == 0):
  pl.plot(time_array, global_data)
  pl.xlabel('Time')
  pl.ylabel(r'$MAX(\delta \rho(x))$')
  pl.savefig('plot.png')
