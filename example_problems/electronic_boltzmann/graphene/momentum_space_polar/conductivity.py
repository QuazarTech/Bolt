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
pl.rcParams['figure.figsize']  = 12, 7.5
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

source_start = params.contact_start
source_end   = params.contact_end

drain_start  = params.contact_start
drain_end    = params.contact_end

source_indices =  (q2 > source_start) & (q2 < source_end)
drain_indices  =  (q2 > drain_start)  & (q2 < drain_end )

# Left needs to be near source, right sensor near drain
sensor_1_left_start = 0.0 # um
sensor_1_left_end   = 1.0 # um

sensor_1_right_start = 0.0 # um
sensor_1_right_end   = 1.0 # um

sensor_1_left_indices  = (q2 > sensor_1_left_start ) & (q2 < sensor_1_left_end)
sensor_1_right_indices = (q2 > sensor_1_right_start) & (q2 < sensor_1_right_end)

filepath = \
'/home/mchandra/gitansh/bolt/example_problems/electronic_boltzmann/graphene/L_2.5_1.0_tau_ee_inf_tau_eph_0.5_DC/dumps'
moment_files 		  = np.sort(glob.glob(filepath+'/moment*.h5'))
lagrange_multiplier_files = \
        np.sort(glob.glob(filepath+'/lagrange_multipliers*.h5'))

dt = params.dt
dump_interval = params.dump_steps

#sensor_1_left_array = []
#sensor_1_right_array = []
#sensor_1_signal_array = []

print("Reading sensor signal...")

file_number = moment_files.size-1
dump_file = moment_files[file_number]

x_index = 24

print ("x index : ", x_index, "x : ", q1[x_index], ", ", q1[-x_index-1])
h5f  = h5py.File(dump_file, 'r')
moments = np.swapaxes(h5f['moments'][:], 0, 1)
h5f.close()

density = moments[:, :, 0]

sensor_1_left   = np.mean(density[x_index,  sensor_1_left_indices])
#sensor_1_left_array.append(sensor_1_left)

sensor_1_right  = np.mean(density[-x_index-1, sensor_1_right_indices])
#sensor_1_right_array.append(sensor_1_right)

sensor_1_signal = sensor_1_left - sensor_1_right
print ("Voltage = ", sensor_1_signal)
#sensor_1_signal_array.append(sensor_1_signal)
    
#pl.rcParams['figure.figsize']  = 12, 7.5

#pl.plot(q2, sensor_1_left, alpha = 0.5)
#pl.plot(q2, sensor_1_right, alpha = 0.5)
#pl.plot(q1, sensor_1_signal_array, alpha = 0.5)
#pl.axhline(0, color='black', linestyle='--')

#pl.legend(['$V$'], loc=1)
#pl.xlabel(r'$y\;(\mu m)$')
#pl.ylabel('Density')
#pl.ylim([6430, 6450])
#pl.ylim([-1.1, 1.1])

#pl.suptitle('$\\tau_\mathrm{mc} = \infty$ ps, $\\tau_\mathrm{mr} = 0.5$ ps')
#pl.savefig('images/iv' + '.png')
pl.clf()
#    

