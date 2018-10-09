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
sensor_1_left_start = 1.5 # um
sensor_1_left_end   = 2.5 # um

sensor_1_right_start = 1.5 # um
sensor_1_right_end   = 2.5 # um

sensor_1_left_indices  = (q2 > sensor_1_left_start ) & (q2 < sensor_1_left_end)
sensor_1_right_indices = (q2 > sensor_1_right_start) & (q2 < sensor_1_right_end)

sensor_2_left_start = 6.5 # um
sensor_2_left_end   = 7.5 # um

sensor_2_right_start = 6.5 # um
sensor_2_right_end   = 7.5 # um

sensor_2_left_indices  = (q2 > sensor_2_left_start ) & (q2 < sensor_2_left_end)
sensor_2_right_indices = (q2 > sensor_2_right_start) & (q2 < sensor_2_right_end)

filepath = \
'/home/mchandra/gitansh/bolt/example_problems/electronic_boltzmann/graphene/L_1.0_2.5_tau_ee_inf_tau_eph_5.0_DC/dumps'
moment_files 		  = np.sort(glob.glob(filepath+'/moment*.h5'))
lagrange_multiplier_files = \
        np.sort(glob.glob(filepath+'/lagrange_multipliers*.h5'))

dt = params.dt
dump_interval = params.dump_steps

sensor_1_signal_array = []
#sensor_2_signal_array = []
#print("Reading sensor signal...")
print("Loading data...")
density = []
edge_density = []
for file_number, dump_file in enumerate(moment_files):

    print("File number = ", file_number)
    h5f  = h5py.File(dump_file, 'r')
    moments = np.swapaxes(h5f['moments'][:], 0, 1)
    h5f.close()

    density.append(moments[:, :, 0])
    edge_density.append(density[file_number][0, sensor_1_left_indices])

density = np.array(density)
edge_density = np.array(edge_density)

mean_density = np.mean(density)
max_density  = np.max(density)
min_density  = np.min(density)

np.savetxt("edge_density.txt", edge_density - mean_density)
np.savetxt("q2_edge.txt", q2[sensor_1_left_indices])

#for file_number in yt.parallel_objects(range(density.shape[0])):
#
#    print("File number = ", file_number)
##    source = np.mean(density[0, source_indices])
##    drain  = np.mean(density[-1, drain_indices])
##
##    sensor_1_left   = np.mean(density[0,  sensor_1_left_indices] )
##    sensor_1_right  = np.mean(density[-1, sensor_1_right_indices])
##
##    sensor_1_signal = sensor_1_left - sensor_1_right
##    sensor_1_signal_array.append(sensor_1_signal)
#    
##    print ('density[0, sensor_1_left_indices] - np.mean(density) : ',
##            (density[0, sensor_1_left_indices] - np.mean(density)).shape)
#
##    pl.plot(q2[sensor_1_left_indices], 
##            density[file_number][0, sensor_1_left_indices] - mean_density,
##           )
#    pl.semilogy(q2[sensor_1_left_indices], 
#                density[file_number][0, sensor_1_left_indices],
#               )
#    #pl.title(r'Time = ' + "%.2f"%(time_array[file_number]) + " ps")
#    pl.title(r'Time = ' + "%.2f"%(file_number*dt*dump_interval) + " ps")
#
#    pl.xlim([sensor_1_left_start, sensor_1_left_end])
#    #pl.ylim([min_density-mean_density, max_density-mean_density])
#    #pl.ylim([0., np.log(max_density)])
#    
#    #pl.gca().set_aspect('equal')
#    #pl.xlabel(r'$x\;(\mu \mathrm{m})$')
#    #pl.ylabel(r'$y\;(\mu \mathrm{m})$')
#    
#    #pl.suptitle('$\\tau_\mathrm{mc} = \infty$ ps, $\\tau_\mathrm{mr} = 3.0$ ps')
#    #pl.savefig('images/dump_' + '%06d'%file_number + '.png')
#    pl.savefig('images/density_' + '%06d'%file_number + '.png')
#    pl.clf()
    
#    sensor_2_left   = np.mean(density[0,  sensor_2_left_indices] )
#    sensor_2_right  = np.mean(density[-1, sensor_2_right_indices])
#
#    sensor_2_signal = sensor_2_left - sensor_2_right
#    sensor_2_signal_array.append(sensor_2_signal)

#time_array = np.loadtxt("dump_time_array.txt")
#AC_freq = 1./100
#input_signal_array = np.sin(2.*np.pi*AC_freq*time_array)
#sensor_1_signal_array = np.array(sensor_1_signal_array)
##sensor_2_signal_array = np.array(sensor_2_signal_array)
#half_time = (int)(time_array.size/2)
#
#input_normalized = \
#        input_signal_array/np.max(np.abs(input_signal_array[half_time:]))
#sensor_1_normalized = \
#    sensor_1_signal_array/np.max(np.abs(sensor_1_signal_array[half_time:]))
#sensor_2_normalized = \
#    sensor_2_signal_array/np.max(np.abs(sensor_2_signal_array[half_time:]))
#
#
## Calculate the phase difference between input_signal_array and sensor_normalized
## Code copied from :
## https:/stackoverflow.com/questions/6157791/find-phase-difference-between-two-inharmonic-waves
#
##corr = correlate(input_normalized, sensor_normalized)
##nsamples = input_normalized.size
##time_corr = time_array[half_time:]
##dt_corr = np.linspace(-time_corr[-1] + time_corr[0],
##                            time_corr[-1] - time_corr[0], 2*nsamples-1)
##time_shift = dt_corr[corr.argmax()]
##
###Force the phase shift to be in [-pi:pi]
##period = 1./AC_freq
##phase_diff = 2*np.pi*(((0.5 + time_shift/period) % 1.0) - 0.5)
#
#pl.rcParams['figure.figsize']  = 12, 7.5
#
#pl.plot(time_array, input_signal_array)
#pl.plot(time_array, sensor_1_normalized)
##pl.plot(time_array, sensor_2_normalized)
#pl.axhline(0, color='black', linestyle='--')
#
#pl.legend(['Source $I(t)$', 'Measured $V_{1}(t)$', 'Measured $V_{2}(t)$'], loc=1)
##pl.text(135, 1.14, '$\phi : %.3f \; rad$' %phase_diff)
#pl.xlabel(r'Time (ps)')
#pl.xlim([0, 200])
#pl.ylim([-1.1, 1.1])
#
#pl.suptitle('$\\tau_\mathrm{mc} = \infty$ ps, $\\tau_\mathrm{mr} = 3.0$ ps')
#pl.savefig('images/iv' + '.png')
#pl.clf()
    

