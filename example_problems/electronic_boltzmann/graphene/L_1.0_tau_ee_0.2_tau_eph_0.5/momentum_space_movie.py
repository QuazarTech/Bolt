import arrayfire as af
import numpy as np
from scipy.signal import correlate
import glob
import h5py
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
matplotlib.use('agg')
import pylab as pl
import yt
yt.enable_parallelism()

import petsc4py, sys; petsc4py.init(sys.argv)
from petsc4py import PETSc

from bolt.lib.physical_system import physical_system

from bolt.lib.nonlinear_solver.nonlinear_solver \
    import nonlinear_solver
from bolt.lib.nonlinear_solver.EM_fields_solver.electrostatic \
    import compute_electrostatic_fields

import domain
import boundary_conditions
import params
import initialize

import bolt.src.electronic_boltzmann.advection_terms as advection_terms

import bolt.src.electronic_boltzmann.collision_operator \
    as collision_operator

import bolt.src.electronic_boltzmann.moment_defs as moment_defs

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 8, 8
pl.rcParams['figure.dpi']      = 100
pl.rcParams['image.cmap']      = 'jet'
pl.rcParams['lines.linewidth'] = 1.5
pl.rcParams['font.family']     = 'serif'
pl.rcParams['font.weight']     = 'bold'
pl.rcParams['font.size']       = 25
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

N_q1 = domain.N_q1
N_q2 = domain.N_q2

q1 = domain.q1_start + (0.5 + np.arange(N_q1)) * (domain.q1_end - domain.q1_start)/N_q1
q2 = domain.q2_start + (0.5 + np.arange(N_q2)) * (domain.q2_end - domain.q2_start)/N_q2

q2_meshgrid, q1_meshgrid = np.meshgrid(q2, q1)

N_p1 = domain.N_p1
N_p2 = domain.N_p2

p1 = domain.p1_start + (0.5 + np.arange(N_p1)) * (domain.p1_end - domain.p1_start)/N_p1
p2 = domain.p2_start + (0.5 + np.arange(N_p2)) * (domain.p2_end - domain.p2_start)/N_p2

p2_meshgrid, p1_meshgrid = np.meshgrid(p2, p1)

filepath = \
'/home/mchandra/gitansh/bolt/example_problems/electronic_boltzmann/graphene/L_5.0_tau_ee_0.2_tau_eph_0.5/dumps'
moment_files 		        = np.sort(glob.glob(filepath+'/moment*.h5'))
lagrange_multiplier_files   = \
        np.sort(glob.glob(filepath+'/lagrange_multipliers*.h5'))
distribution_function_files = np.sort(glob.glob(filepath+'/f_*.h5'))

dt = params.dt
dump_interval = params.dump_steps

time_array = np.loadtxt("dump_time_array.txt")

h5f  = h5py.File(distribution_function_files[0], 'r')
dist_func_background = h5f['distribution_function'][:]
h5f.close()

q1_position = 30
q2_position = 90

for file_number, dump_file in yt.parallel_objects(enumerate(distribution_function_files)):
    
    print("file number = ", file_number, "of ", distribution_function_files.size)

    h5f  = h5py.File(distribution_function_files[file_number], 'r')
    dist_func = h5f['distribution_function'][:]
    h5f.close()


    f_at_desired_q = np.reshape((dist_func - dist_func_background)[q2_position, q1_position, :],
                                [N_p1, N_p2]
                               )
    pl.contourf(p1_meshgrid, p2_meshgrid, f_at_desired_q, 100, cmap='bwr')
    pl.title(r'Time = ' + "%.2f"%(time_array[file_number]) + " ps")
    pl.xlabel('$p_x$')
    pl.ylabel('$p_y$')
    pl.gca().set_aspect('equal')
    pl.savefig('images/dist_func_' + '%06d'%file_number + '.png')
    pl.clf()

