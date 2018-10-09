import arrayfire as af
import numpy as np
from scipy.signal import correlate
from scipy.interpolate import interp2d
import scipy.fftpack
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

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

N_samples = 1000
step_size = np.pi*2/N_samples
radius = np.arange(0, domain.p1_end, 0.001)
theta = np.linspace(0, N_samples*step_size, N_samples)

filepath = \
'/home/mchandra/gitansh/bolt/example_problems/electronic_boltzmann/graphene/L_1.0_2.5_tau_ee_inf_tau_eph_5.0_DC_ultra_high_res/dumps'
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

h5f  = h5py.File(distribution_function_files[-1], 'r')
dist_func = h5f['distribution_function'][:]
h5f.close()

file_number = moment_files.size-1


radial_PSD_array = np.sin(theta)

pl.subplot(2, 1, 1)
pl.plot(theta, radial_PSD_array, '-o')

pl.xlabel('$\\theta$')
pl.ylabel('f')
        
pl.subplot(2, 1, 2)
xf       = 2*np.pi*np.linspace(0.0, 1.0/(2.0*step_size), N_samples/2)
yf  = scipy.fftpack.fft(radial_PSD_array)

pl.plot(xf, 2.0/N_samples * np.abs(yf[:int(N_samples/2)]))
#pl.title  ("$\mathrm{Radial PSD}$")
pl.xlabel ('$\mathrm{k}$')
pl.xlim([0, 500])

pl.tight_layout()
pl.savefig('images/test.png')
pl.clf()


avg_PSD_array = []
N_1 = 12
N_2 = 30
for index_1 in range(N_1):
    for index_2 in range(N_2):

        print('Index : ', index_1, index_2)

        q1_position = int(domain.N_q1*((index_1/N_1)+(1/(2*N_1))))
        q2_position = int(domain.N_q2*((index_2/N_2)+(1/(2*N_2))))
        
        a = np.max((dist_func - dist_func_background)[q2_position, q1_position, :])
        b = np.abs(np.min((dist_func - dist_func_background)[q2_position, q1_position, :]))
        norm_factor = np.maximum(a, b)
        f_at_desired_q = \
        np.reshape((dist_func-dist_func_background)[q2_position, q1_position,:], [N_p1, N_p2])/norm_factor

        f = interp2d(p1, p2, f_at_desired_q, kind='cubic')
        
        radial_PSD_array = []
        f_vs_r_array = []
        for angle in theta:
            print ('Angle : ', angle)
            #summation = 0
            f_vs_r = []
            for r in radius:
                x, y = pol2cart(r, angle)
                f_vs_r.append(f(x, y))
            #pl.plot(radius, f_vs_r)
            radial_PSD_array.append(np.mean(f_vs_r))

        pl.subplot(2, 1, 1)
        pl.plot(theta, radial_PSD_array, '-o', alpha = 1.0, linewidth = 1)

        pl.xlabel('$\mathrm{\\theta}$')
        pl.ylabel('$\mathrm{f(\\theta)}$')
        
        pl.subplot(2, 1, 2)
        xf  = 2*np.pi*np.linspace(0.0, 1.0/(2.0*step_size), N_samples/2)
        yf  = scipy.fftpack.fft(radial_PSD_array)

        pl.loglog(xf, 2.0/N_samples * np.abs(yf[:int(N_samples/2)]), alpha = 0.5, linewidth = 2)
        #pl.title  ("$\mathrm{Radial PSD}$")
        pl.xlabel ('$\mathrm{k_{\\theta}}$')
        pl.ylabel ('$\mathrm{\hat{f}(k_{\\theta})}$')

        pl.loglog(xf[10:], xf[10:]**(-1.4), linestyle='--', color='black')
        pl.xlim(xmax=200)
        pl.ylim(ymin=1e-4)

        pl.tight_layout()
        pl.savefig('images/f_vs_theta_at_a_point_%d_%d.png'%(index_1, index_2))
        pl.clf()

        avg_PSD_array.append(radial_PSD_array)
        np.savetxt('f_vs_thera_at_a_point_%d_%d.txt'%(index_1, index_2), radial_PSD_array)

#f_spatial_avg = np.reshape(np.mean(dist_func-dist_func_background, axis=(0,1)), [N_p1, N_p2])

pl.subplot(2, 1, 1)
pl.plot(theta, avg_PSD_array[0], alpha = 1.0, linewidth = 1, color = 'C0')
    
pl.xlabel('$\mathrm{\\theta}$')
pl.ylabel('$\mathrm{f(\\theta)}$')
    
pl.subplot(2, 1, 2)
for PSD in avg_PSD_array:
    xf = 2*np.pi*np.linspace(0.0, 1.0/(2.0*step_size), N_samples/2)
    yf = scipy.fftpack.fft(PSD)
    
    pl.loglog(xf, 2.0/N_samples * np.abs(yf[:int(N_samples/2)]), alpha = 0.2, linewidth = 2, color = 'C0')

pl.loglog(xf[10:], xf[10:]**(-1.4), linestyle='--', color='black')
pl.xlim(xmax = 200)
pl.ylim(ymin=1e-4)

pl.xlabel ('$\mathrm{k_{\\theta}}$')
pl.ylabel ('$\mathrm{\hat{f}(k_{\\theta})}$')
    
pl.tight_layout()
pl.savefig('images/f_vs_theta_overplot.png')
pl.clf()


avg_PSD_array = np.array(avg_PSD_array)
print (avg_PSD_array.shape)
avg_PSD = np.mean(avg_PSD_array, axis = 0)

pl.subplot(2, 1, 1)
pl.plot(theta, avg_PSD, '-o')

pl.xlabel('$\mathrm{\\theta}$')
pl.ylabel('$\mathrm{f(\\theta)}$')

pl.subplot(2, 1, 2)
xf = 2*np.pi*np.linspace(0.0, 1.0/(2.0*step_size), N_samples/2)
yf = scipy.fftpack.fft(avg_PSD)

pl.loglog(xf, 2.0/N_samples * np.abs(yf[:int(N_samples/2)]))
pl.xlabel ('$\mathrm{k_{\\theta}}$')
pl.ylabel ('$\mathrm{\hat{f}(k_{\\theta})}$')

pl.tight_layout()
pl.savefig('images/f_vs_theta_spatial_avg.png')
pl.clf()


# Real space plot
h5f  = h5py.File(moment_files[file_number], 'r')
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

pl.subplot(1,2,1)
pl.contourf(q1_meshgrid, q2_meshgrid, density, 100, cmap='bwr')
pl.title(r'Time = ' + "%.2f"%(time_array[file_number]) + " ps")
pl.streamplot(q1, q2, 
              vel_drift_x, vel_drift_y,
              density=2, color='blue',
              linewidth=0.7, arrowsize=1
             )

pl.xlim([domain.q1_start, domain.q1_end])
pl.ylim([domain.q2_start, domain.q2_end])

pl.gca().set_aspect('equal')
pl.xlabel(r'$x\;(\mu \mathrm{m})$')
pl.ylabel(r'$y\;(\mu \mathrm{m})$')

pl.subplot(1,2,2)
pl.contourf(q1_meshgrid, q2_meshgrid, density, 100, cmap='bwr')
pl.title(r'Time = ' + "%.2f"%(time_array[file_number]) + " ps")
for index_1 in range(N_1):
    for index_2 in range(N_2):
        q1_position = int(domain.N_q1*((index_1/N_1)+(1/(2*N_1))))
        q2_position = int(domain.N_q2*((index_2/N_2)+(1/(2*N_2))))
        txt = "(%d, %d)"%(index_1, index_2)
        pl.plot(q1[q1_position], q2[q2_position], color = 'k', marker = 'o',
                markersize = 3)
        pl.text(q1[q1_position], q2[q2_position], txt, fontsize=8)

pl.xlim([domain.q1_start, domain.q1_end])
#pl.ylim([domain.q2_start, domain.q2_end])
