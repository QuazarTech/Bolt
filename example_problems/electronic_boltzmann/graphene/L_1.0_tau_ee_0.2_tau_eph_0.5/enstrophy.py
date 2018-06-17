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

filepath = \
'/home/mchandra/gitansh/bolt/example_problems/electronic_boltzmann/graphene/L_1.0_tau_ee_0.2_tau_eph_0.5/dumps'
moment_files 		  = np.sort(glob.glob(filepath+'/moment*.h5'))
lagrange_multiplier_files = \
        np.sort(glob.glob(filepath+'/lagrange_multipliers*.h5'))

dt = params.dt
dump_interval = params.dump_steps

kinetic_energy_array = []
enstrophy_array      = []
print("Reading kinetic energy and enstrophy signals...")
for file_number, dump_file in enumerate(moment_files):

    h5f  = h5py.File(dump_file, 'r')
    moments = np.swapaxes(h5f['moments'][:], 0, 1)
    h5f.close()

    density = moments[:, :, 0]
    j_x     = moments[:, :, 1]
    j_y     = moments[:, :, 2]

    h5f  = h5py.File(lagrange_multiplier_files[file_number], 'r')
    lagrange_multipliers = h5f['lagrange_multipliers'][:]
    h5f.close()

    mu    = lagrange_multipliers[:, :, 0]
    mu_ee = lagrange_multipliers[:, :, 1]
    T_ee  = lagrange_multipliers[:, :, 2]
    vel_drift_x = lagrange_multipliers[:, :, 3]
    vel_drift_y = lagrange_multipliers[:, :, 4]
    
    dq1 = (domain.q1_end - domain.q1_start)/domain.N_q1
    dq2 = (domain.q2_end - domain.q2_start)/domain.N_q2

    dvx_dx, dvx_dy = np.gradient(j_x/density, dq1, dq2)
    dvy_dx, dvy_dy = np.gradient(j_y/density, dq1, dq2)

    vorticity = dvy_dx - dvx_dy
    
    kinetic_energy = 0.5 * np.sum(vel_drift_x**2 + vel_drift_y**2) * dq1 * dq2
    kinetic_energy_array.append(kinetic_energy)

    enstrophy = np.sum(vorticity**2) * dq1 * dq2
    enstrophy_array.append(enstrophy)

time_array = np.loadtxt("dump_time_array.txt")
half_time = (int)(time_array.size/2)

kinetic_energy_normalized = \
        kinetic_energy_array/np.max(np.abs(kinetic_energy_array[half_time:]))
enstrophy_normalized = \
    enstrophy_array/np.max(np.abs(enstrophy_array[half_time:]))


pl.plot(time_array, kinetic_energy_normalized)
pl.plot(time_array, enstrophy_normalized)
pl.axhline(0, color='black', linestyle='--')

pl.legend(['Kinetic Energy', 'Enstrophy'], loc=1)
pl.xlabel(r'Time (ps)')
pl.xlim([0, 200])
pl.ylim([-0.1, 1.1])

pl.suptitle('$\\tau_\mathrm{mc} = 0.2$ ps, $\\tau_\mathrm{mr} = 1.0$ ps')
pl.savefig('vorticity_images/iv' + '.png')
pl.clf()
    

