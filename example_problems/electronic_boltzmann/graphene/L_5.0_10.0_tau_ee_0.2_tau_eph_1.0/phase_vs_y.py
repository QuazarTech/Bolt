import arrayfire as af
import numpy as np
from scipy.signal import correlate
from scipy.optimize import curve_fit
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

def sin_curve_fit(t, A, tau):
        return A*np.sin(2*np.pi*AC_freq*(t + tau ))

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
sensor_1_left_start = 5.5 # um
sensor_1_left_end   = 10.0 # um

sensor_1_right_start = 5.5 # um
sensor_1_right_end   = 10.0 # um

sensor_1_left_indices  = (q2 > sensor_1_left_start ) & (q2 < sensor_1_left_end)
sensor_1_right_indices = (q2 > sensor_1_right_start) & (q2 < sensor_1_right_end)

filepath = \
    '/home/mchandra/gitansh/bolt/example_problems/electronic_boltzmann/graphene/L_2.5_tau_ee_inf_tau_eph_2.5/dumps'
    
AC_freq        = 1./100.0
time_period    = 1/AC_freq
t_final        = params.t_final
transient_time = t_final/2.

time         = np.loadtxt(filepath + "/../dump_time_array.txt")
edge_density = np.loadtxt(filepath + "/../edge_density.txt")
q2           = np.loadtxt(filepath + "/../q2_edge.txt")

N_spatial = edge_density.shape[1]

transient_index = int((transient_time/t_final)*time.size)
    
drive = np.sin(2*np.pi*AC_freq*time)
nsamples = drive.size
dt_corr = np.linspace(-time[-1] + time[0],\
        time[-1] - time[0], 2*nsamples-1)

# Discarding transients
q = q2.size/2
time_half = time[transient_index:]
drive_half = drive[transient_index:]

# Plotting signals at edge
norm_0 = np.max(edge_density[transient_index:, 0])
norm_1 = np.max(edge_density[transient_index:, -1])

pl.plot(time, drive, color='black', linestyle='--')
pl.ylim([-1.1, 1.1])
pl.xlim([0,200])
pl.xlabel('$\mathrm{Time\;(s)}$')

for i in range(N_spatial):
    norm_i = np.max(edge_density[transient_index:, i])
    pl.plot(time, edge_density[:, i]/norm_i)

pl.savefig('images/signals.png')
pl.clf()

phase_shift_corr_array = []
phase_shift_fitting_array = []\

for i in range(N_spatial):
    print ('index : ', i)
    signal_1 = edge_density[:, i]
    norm_1 = np.max(signal_1[transient_index:])
    signal_1_normalized = signal_1/norm_1
        
    # Calculate phase_shifts using scipy.correlate
    corr = correlate(drive, signal_1_normalized)
    time_shift_corr = dt_corr[corr.argmax()]
    phase_shift_corr  = 2*np.pi*(((0.5 + time_shift_corr/time_period) % 1.0) - 0.5)

    # Calculate phase_shifts using scipy.curve_fit
    popt, pcov = curve_fit(sin_curve_fit, time[transient_index:],\
                signal_1_normalized[transient_index:])
    time_shift_fitting = popt[1]%(time_period/2.0)
    phase_shift_fitting  = 2*np.pi*(((0.5 + time_shift_fitting/time_period) % 1.0) - 0.5)

    phase_shift_corr_array.append(phase_shift_corr)
    phase_shift_fitting_array.append(phase_shift_fitting)

phase_shift_corr_array = np.array(phase_shift_corr_array)
phase_shift_fitting_array = np.array(phase_shift_fitting_array)

# Plot
pl.ylabel('$\mathrm{\phi}$')
pl.xlabel('$\mathrm{y\ \mu m}$')

pl.plot(q2, phase_shift_corr_array, '-o', label='$\mathrm{corr}$')
pl.plot(q2, phase_shift_fitting_array, '-o', label='$\mathrm{fit}$')

pl.title('$\mathrm{2.5 \\times 10,\ \\tau_{ee} = \infty,\ \\tau_{eph} = 2.5}$')
pl.legend(loc='best')

#pl.axvspan(sensor_1_left_start, sensor_1_left_end, color = 'k', alpha = 0.1)
#pl.axvspan(sensor_2_left_start, sensor_2_left_end, color = 'k', alpha = 0.1)

pl.savefig('images/phase_vs_y.png')
pl.clf()

