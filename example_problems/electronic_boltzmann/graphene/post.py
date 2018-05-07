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
#pl.rcParams['figure.figsize']  = 8, 7.5
pl.rcParams['figure.figsize']  = 8, 8
#pl.rcParams['figure.figsize']  = 17, 9.5
pl.rcParams['figure.dpi']      = 100
pl.rcParams['image.cmap']      = 'jet'
pl.rcParams['lines.linewidth'] = 1.5
#pl.rcParams['lines.linewidth'] = 3
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
#N_q1 = 120
#N_q2 = 240

q1 = domain.q1_start + (0.5 + np.arange(N_q1)) * (domain.q1_end - domain.q1_start)/N_q1
q2 = domain.q2_start + (0.5 + np.arange(N_q2)) * (domain.q2_end - domain.q2_start)/N_q2

q2_meshgrid, q1_meshgrid = np.meshgrid(q2, q1)

source_start = params.contact_start
source_end   = params.contact_end

drain_start  = params.contact_start
drain_end    = params.contact_end

#source_start = 3.5; source_end = 4.5
#drain_start  = 5.5; drain_end  = 6.5

source_indices =  (q2 > source_start) & (q2 < source_end)
drain_indices  =  (q2 > drain_start)  & (q2 < drain_end )

sensor_1_left_start = 8.5 # um
sensor_1_left_end   = 9.5 # um

sensor_1_right_start = 8.5 # um
sensor_1_right_end   = 9.5 # um

# Left needs to be near source, right sensor near drain
#sensor_1_left_start = 1.5 # um
#sensor_1_left_end   = 2.5 # um

#sensor_1_right_start = 7.5 # um
#sensor_1_right_end   = 8.5 # um

sensor_1_left_indices  = (q2 > sensor_1_left_start ) & (q2 < sensor_1_left_end)
sensor_1_right_indices = (q2 > sensor_1_right_start) & (q2 < sensor_1_right_end)

sensor_2_left_start = 6.5 # um
sensor_2_left_end   = 7.5 # um

sensor_2_right_start = 6.5 # um
sensor_2_right_end   = 7.5 # um

sensor_2_left_indices  = (q2 > sensor_2_left_start ) & (q2 < sensor_2_left_end)
sensor_2_right_indices = (q2 > sensor_2_right_start) & (q2 < sensor_2_right_end)

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

#filepath = \
#'/home/mchandra/bolt/example_problems/electronic_boltzmann/graphene/geom_1/DC/tau_D_50_tau_ee_0.2'
#dump_file= np.sort(glob.glob(filepath+'/moment*.h5'))[-1]
#
#h5f  = h5py.File(dump_file, 'r')
#moments = np.swapaxes(h5f['moments'][:], 0, 1)
#h5f.close()
#
#density = moments[:, :, 0]
#np.savetxt('paper_plots/density_tau_D_50_tau_ee_0.2.txt', density)
#np.savetxt('paper_plots/q2_DC_tau_D_50_tau_ee_0.2.txt', q2)
##pl.plot(q2[q2>source_end], density[0, q2>source_end]-np.mean(density))
#
#filepath = \
#'/home/mchandra/bolt/example_problems/electronic_boltzmann/graphene/geom_1/DC/tau_D_5_tau_ee_0.2'
#dump_file= np.sort(glob.glob(filepath+'/moment*.h5'))[-1]
#
#
#h5f  = h5py.File(dump_file, 'r')
#moments = np.swapaxes(h5f['moments'][:], 0, 1)
#h5f.close()
#
#density = moments[:, :, 0]
#np.savetxt('paper_plots/density_tau_D_5_tau_ee_0.2.txt', density)
#np.savetxt('paper_plots/q2_DC_tau_D_5_tau_ee_0.2.txt', q2)
#
#filepath = \
#'/home/mchandra/bolt/example_problems/electronic_boltzmann/graphene/geom_1/DC/tau_D_10_tau_ee_0.2'
#dump_file= np.sort(glob.glob(filepath+'/moment*.h5'))[-1]
#
#N_q1 = 120
#N_q2 = 240
#
#q1 = domain.q1_start + (0.5 + np.arange(N_q1)) * (domain.q1_end - domain.q1_start)/N_q1
#q2 = domain.q2_start + (0.5 + np.arange(N_q2)) * (domain.q2_end - domain.q2_start)/N_q2
#
#h5f  = h5py.File(dump_file, 'r')
#moments = np.swapaxes(h5f['moments'][:], 0, 1)
#h5f.close()
#
#density = moments[:, :, 0]
#np.savetxt('paper_plots/density_tau_D_10_tau_ee_0.2.txt', density)
#np.savetxt('paper_plots/q2_DC_tau_D_10_tau_ee_0.2.txt', q2)


#pl.plot(q2[q2>source_end], density[0, q2>source_end]-np.mean(density))
#pl.axhline(0, color='black', linestyle='--')
#pl.legend(['$\\tau_{ee}=0.2$ ps, $\\tau_{e-ph}=50$ ps',
#           '$\\tau_{ee}=0.2$ ps, $\\tau_{e-ph}=10$ ps'], loc='lower right')
#pl.xlabel(r'$x\;(\mu \mathrm{m})$')
#pl.ylabel(r'$R\; (\mathrm{a.u.})$')
#pl.xlim(xmin=(source_end+0.1))
#pl.savefig('paper_plots/DC.png')

filepath = \
'/home/mchandra/bolt/example_problems/electronic_boltzmann/graphene/dumps_tau_D_1_tau_ee_0.2_movie'
#'/home/mchandra/bolt/example_problems/electronic_boltzmann/graphene/geom_1/55_GHz/tau_D_5_tau_ee_0.2'
#'/home/mchandra/bolt/example_problems/electronic_boltzmann/graphene/dumps_AC_10_Ghz_tau_D_10_tau_ee_1_geom_2/'
#'/home/mchandra/bolt/example_problems/electronic_boltzmann/graphene/dumps_tau_D_2_tau_ee_1_AC/'
moment_files 		  = np.sort(glob.glob(filepath+'/moment*.h5'))
lagrange_multiplier_files = \
        np.sort(glob.glob(filepath+'/lagrange_multipliers*.h5'))

dt = params.dt
dump_interval = params.dump_steps

sensor_1_signal_array = []
print("Reading sensor signal...")
for file_number, dump_file in enumerate(moment_files):

    h5f  = h5py.File(dump_file, 'r')
    moments = np.swapaxes(h5f['moments'][:], 0, 1)
    h5f.close()

    density = moments[:, :, 0]
    
    source = np.mean(density[0, source_indices])
    drain  = np.mean(density[-1, drain_indices])

    sensor_1_left   = np.mean(density[0,  sensor_1_left_indices] )
    sensor_1_right  = np.mean(density[-1, sensor_1_right_indices])

    sensor_1_signal = sensor_1_left - sensor_1_right

    sensor_1_signal_array.append(sensor_1_signal)

time_array = np.loadtxt("dump_time_array.txt")
AC_freq = 1./100
input_signal_array = np.sin(2.*np.pi*AC_freq*time_array)
sensor_1_signal_array = np.array(sensor_1_signal_array)
half_time = (int)(time_array.size/2)
sensor_normalized = \
    sensor_1_signal_array/np.max(np.abs(sensor_1_signal_array[half_time:]))

pl.rcParams['figure.figsize']  = 10, 8
for file_number, dump_file in yt.parallel_objects(enumerate(moment_files)):

    print("file number = ", file_number, "of ", moment_files.size)

    h5f  = h5py.File(dump_file, 'r')
    moments = np.swapaxes(h5f['moments'][:], 0, 1)
    h5f.close()

    gs = gridspec.GridSpec(3, 2)
    pl.subplot(gs[:, 0])

    density = moments[:, :, 0]
    j_x     = moments[:, :, 1]
    j_y     = moments[:, :, 2]
    pl.contourf(q1_meshgrid, q2_meshgrid, density, 100, cmap='bwr')
    pl.title(r'Time = ' + "%.2f"%(time_array[file_number]) + " ps")
    #pl.colorbar()

    h5f  = h5py.File(lagrange_multiplier_files[file_number], 'r')
    lagrange_multipliers = h5f['lagrange_multipliers'][:]
    h5f.close()

    mu    = lagrange_multipliers[:, :, 0]
    mu_ee = lagrange_multipliers[:, :, 1]
    T_ee  = lagrange_multipliers[:, :, 2]
    vel_drift_x = lagrange_multipliers[:, :, 3]
    vel_drift_y = lagrange_multipliers[:, :, 4]
 
#    pl.streamplot(q1[(int)(N_q1/2):], q2, 
#            vel_drift_x[:, (int)(N_q1/2):], vel_drift_y[:, (int)(N_q1/2):],
#                  density=2, color='blue',
#                  linewidth=0.7, arrowsize=1
#                 )
    pl.streamplot(q1, q2, 
                  vel_drift_x, vel_drift_y,
                  density=2, color='blue',
                  linewidth=0.7, arrowsize=1
                 )
#    pl.streamplot(q1, q2, 
#                  vel_drift_x, vel_drift_y,
#                  density=3, color='blue',
#                  linewidth=0.8, arrowsize=1.1
#                 )
    pl.xlim([domain.q1_start, domain.q1_end])
    pl.ylim([domain.q2_start, domain.q2_end])
    #pl.ylim([0, 5])
    pl.gca().set_aspect('equal')
    pl.xlabel(r'$x\;(\mu \mathrm{m})$')
    pl.ylabel(r'$y\;(\mu \mathrm{m})$')

    pl.gca().annotate("+", xy=(-0.07, .9), xycoords=("axes fraction"),
            ha="center", va="center", size=30,
            bbox=dict(fc="white"))
    pl.gca().annotate("-", xy=(1.05, .9), xycoords=("axes fraction"),
            ha="center", va="center", size=30,
            bbox=dict(fc="white", pad=6.5))


    pl.subplot(gs[1, 1])

    pl.plot(time_array, input_signal_array)
    pl.plot(time_array, sensor_normalized)
    pl.axhline(0, color='black', linestyle='--')
    pl.axvline(time_array[file_number], color='black', alpha=0.75)
    pl.legend(['Source $I(t)$', 'Measured $V(t)$'], loc=(0.04, 1.125))
    pl.xlabel(r'Time (ps)')
    pl.xlim([100, 200])
    pl.ylim([-1.1, 1.1])
    

    pl.suptitle('$\\tau_\mathrm{mc} = 0.2$ ps, $\\tau_\mathrm{mr} = 1$ ps')
    #pl.tight_layout()
    pl.savefig('images/dump_' + '%06d'%file_number + '.png')
    #pl.savefig('paper_plots/DC.png')
    pl.clf()


#time_array            = []
#input_signal_array    = []
#sensor_1_signal_array = []
#sensor_2_signal_array = []
#for file_number, dump_file in enumerate(moment_files):
#
#    print("file number = ", file_number, "of ", moment_files.size)
#
#    h5f  = h5py.File(dump_file, 'r')
#    moments = np.swapaxes(h5f['moments'][:], 0, 1)
#    h5f.close()
#
#    density = moments[:, :, 0]
#    
#    source = np.mean(density[0, source_indices])
#    drain  = np.mean(density[-1, drain_indices])
#
#    sensor_1_left   = np.mean(density[0,  sensor_1_left_indices] )
#    sensor_1_right  = np.mean(density[-1, sensor_1_right_indices])
#    #sensor_1_right  = np.mean(density[0, sensor_1_right_indices])
#
#    sensor_2_left   = np.mean(density[0,  sensor_2_left_indices] )
#    sensor_2_right  = np.mean(density[-1, sensor_2_right_indices])
#
#    #sensor_1_left  = density[0,  q2>source_end]
#    #sensor_1_right = density[-1, q2>source_end] 
#
#    input_signal    = source        - drain
#    sensor_1_signal = sensor_1_left - sensor_1_right
#    sensor_2_signal = sensor_2_left - sensor_2_right
#
#    time_array.append(file_number*dt*dump_interval)
#    input_signal_array.append(input_signal)
#    sensor_1_signal_array.append(sensor_1_signal)
#    sensor_2_signal_array.append(sensor_2_signal)
#
##pl.rcParams['figure.figsize']  = 12, 8
##
#AC_freq = 5.5/100
#time_array = np.array(time_array)
#input_signal_array = np.sin(2.*np.pi*AC_freq*time_array)
#sensor_1_signal_array = np.array(sensor_1_signal_array)
###np.savetxt('drive.txt', input_signal_array)
##np.savetxt('paper_plots/sensor_tau_ee_0.2_tau_D_5.txt', sensor_1_signal_array)
##np.savetxt('time.txt', time_array)
##
##print("sensor.shape = ", sensor_1_signal_array.shape)
##sensor_2_signal_array = np.array(sensor_2_signal_array)
##
#half_time = (int)(time_array.size/2)
#pl.plot(time_array, input_signal_array/np.max(input_signal_array[half_time:]))
#pl.plot(time_array,
#        sensor_1_signal_array/np.max(sensor_1_signal_array[half_time:])
#       )
##pl.plot(time_array,
##        sensor_2_signal_array/np.max(sensor_2_signal_array[half_time:])
##       )
#pl.xlabel(r"Time (ps)")
#pl.ylim([-1.1, 1.1])
#pl.legend(['Input', 'Sensor 1', 'Sensor 2'])
#pl.savefig('paper_plots/IV_55_Ghz_tau_ee_0.2_tau_D_5.png')
##half_time = 0
#input_normalized  = input_signal_array[half_time:]/np.max(input_signal_array[half_time:])
#sensor_normalized = sensor_1_signal_array[half_time:]/np.max(sensor_1_signal_array[half_time:])
#
## Calculate the phase diff. Copied from:
## https://stackoverflow.com/questions/6157791/find-phase-difference-between-two-inharmonic-waves
#corr = correlate(input_normalized, sensor_normalized)
#nsamples = input_normalized.size
#time_corr = time_array[half_time:]
#dt_corr = np.linspace(-time_corr[-1] + time_corr[0], 
#                       time_corr[-1] - time_corr[0], 2*nsamples-1)
#time_shift = dt_corr[corr.argmax()]
#
## force the phase shift to be in [-pi:pi]
#period = 1./AC_freq
#phase_diff = 2*np.pi*(((0.5 + time_shift/period) % 1.0) - 0.5)
#print("density.shape = ", density.shape)
#print("Phase diff = ", phase_diff)

#phase_vs_x = []
#for i in range(sensor_1_signal_array[0, :].size):
#    signal = sensor_1_signal_array[:, i]
#    corr = correlate(input_signal_array, signal)
#    nsamples = input_signal_array.size
#    half_time = 0
#    time_corr = time_array[half_time:]
#    dt_corr = np.linspace(-time_corr[-1] + time_corr[0], 
#                           time_corr[-1] - time_corr[0], 
#                           2*nsamples-1
#                         )
#    time_shift = dt_corr[corr.argmax()]
#
#    # force the phase shift to be in [-pi:pi]
#    period = 1./params.AC_freq
#    phase_diff = 2*np.pi*(((0.5 + time_shift/period) % 1.0) - 0.5)
#    phase_vs_x.append(phase_diff)
#
#phase_vs_x = np.array(phase_vs_x)
#print("phase_vs_x.shape = ", phase_vs_x.shape)
#np.savetxt("paper_plots/phase_vs_x_tau_ee_0.2_tau_D_1.txt", phase_vs_x)
#np.savetxt("paper_plots/q2_tau_ee_0.2_tau_D_1.txt", q2)
#print("density.shape = ", density.shape)


