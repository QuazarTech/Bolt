import arrayfire as af
import numpy as np
import glob
import h5py
import matplotlib
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
pl.rcParams['figure.figsize']  = 12, 7.5
pl.rcParams['figure.dpi']      = 100
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

N_q1 = domain.N_q1
N_q2 = domain.N_q2
N_g  = domain.N_ghost

q1 = domain.q1_start + (0.5 + np.arange(N_q1)) * (domain.q1_end - domain.q1_start)/N_q1
q2 = domain.q2_start + (0.5 + np.arange(N_q2)) * (domain.q2_end - domain.q2_start)/N_q2

q2, q1 = np.meshgrid(q2, q1)

#dump_index = 0
#h5f  = h5py.File('dumps/moments_' + '%06d'%(dump_index) + '.h5', 'r')
#moments = np.swapaxes(h5f['moments'][:], 0, 1)
#h5f.close()
#
#density = moments[:, :, 0]
#j_x     = moments[:, :, 1]
#j_y     = moments[:, :, 2]
#pl.contourf(q1, q2, density, 100)
##pl.title('Time = ' + "%.2f"%(t0))
#pl.axes().set_aspect('equal')
#pl.xlabel(r'$x$')
#pl.ylabel(r'$y$')
#pl.colorbar()
#pl.savefig('images/density' + '.png')
#pl.clf()
#
#h5f  = h5py.File('dumps/lagrange_multipliers_' + '%06d'%(dump_index) + '.h5', 'r')
#lagrange_multipliers = np.swapaxes(h5f['lagrange_multipliers'][:], 0, 1)
#h5f.close()
#
#print("lagrange_multipliers.shape = ", lagrange_multipliers.shape)
#mu    = lagrange_multipliers[:, :, 0]
#mu_ee = lagrange_multipliers[:, :, 1]
#T_ee  = lagrange_multipliers[:, :, 2]
#vel_drift_x = lagrange_multipliers[:, :, 3]
#vel_drift_y = lagrange_multipliers[:, :, 4]
#j_x_prime = density*vel_drift_x
#print("err = ", np.mean(np.abs(j_x_prime - j_x)))

filepath = '/home/mchandra/bolt/example_problems/electronic_boltzmann/graphene/dumps'
moment_files 		  = np.sort(glob.glob(filepath+'/moment*.h5'))
lagrange_multiplier_files = np.sort(glob.glob(filepath+'/lagrange_multipliers*.h5'))

dt = params.dt
dump_interval = params.dump_steps

for file_number, dump_file in yt.parallel_objects(enumerate(moment_files)):

    print("file number = ", file_number, "of ", moment_files.size)

    h5f  = h5py.File(dump_file, 'r')
    moments = np.swapaxes(h5f['moments'][:], 0, 1)
    h5f.close()

    density = moments[:, :, 0]
    j_x     = moments[:, :, 1]
    j_y     = moments[:, :, 2]
    pl.contourf(q1, q2, density, 100, cmap='bwr')
    pl.title(r'Time = ' + "%.2f"%(file_number*dt*dump_interval))
    pl.axes().set_aspect('equal')
    pl.xlabel(r'$x$')
    pl.ylabel(r'$y$')
    pl.colorbar()
    pl.savefig('images/density_' + '%06d'%file_number + '.png')
    pl.clf()

    h5f  = h5py.File(lagrange_multiplier_files[file_number], 'r')
    lagrange_multipliers = np.swapaxes(h5f['lagrange_multipliers'][:], 0, 1)
    h5f.close()

    mu    = lagrange_multipliers[:, :, 0]
    mu_ee = lagrange_multipliers[:, :, 1]
    T_ee  = lagrange_multipliers[:, :, 2]
    vel_drift_x = lagrange_multipliers[:, :, 3]
    vel_drift_y = lagrange_multipliers[:, :, 4]


#h5f  = h5py.File('dumps/fields_000030.h5', 'r')
#fields = np.swapaxes(h5f['fields'][:], 0, 1)
#h5f.close()
#
#E1 = fields[:, :, 0]
#E2 = fields[:, :, 1]
#
#pl.figure(figsize=(20, 7.5))
#pl.subplot(121)
#pl.contourf(q1, q2, E1, 100, cmap='bwr')
#pl.title('$E_1$')
#pl.xlabel(r'$x$')
#pl.ylabel(r'$y$')
#pl.colorbar()
#pl.gca().set_aspect('equal')
#
#pl.subplot(122)
#pl.contourf(q1, q2, E2, 100, cmap='bwr')
#pl.title('$E_2$')
#pl.xlabel(r'$x$')
#pl.ylabel(r'$y$')
#pl.colorbar()
#pl.gca().set_aspect('equal')
#pl.savefig('images/E_fields' + '.png')
#pl.clf()

#for time_index, t0 in enumerate(time):
#    
#    h5f  = h5py.File('dumps/density_' + str(time_index) + '.h5', 'r')
#    moments = np.swapaxes(h5f['moments'][:], 0, 1)
#    h5f.close()
#    
#    n = moments[:, :, 0]
#
#    #pl.contourf(q1, q2, n, np.linspace(0.8, 2.2, 500))
#    pl.contourf(q1, q2, n, 100)
#    pl.title('Time = ' + "%.2f"%(t0))
#    pl.axes().set_aspect('equal')
#    pl.xlabel(r'$x$')
#    pl.ylabel(r'$y$')
#    pl.colorbar()
#    pl.savefig('images/%04d'%time_index + '.png')
#    pl.clf()
