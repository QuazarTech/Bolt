import numpy as np
import h5py
import matplotlib as mpl 
mpl.use('agg')
import pylab as pl

import input_files.domain as domain
import input_files.params as params

# Optimized plot parameters to make beautiful plots:
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

omega = 0

# Defining the functions for the analytical solution:
def n_analytic(q1, q2, t):
    
    n_b = params.density_background

    pert_real_n = 1
    pert_imag_n = 0
    pert_n      = pert_real_n + 1j * pert_imag_n

    n_analytic       = n_b + params.amplitude * pert_n * \
                        np.exp(  1j * (params.k_q1 * q1 + params.k_q2 * q2) 
                               + omega * t
                              ).real

    return(n_analytic)

def v1_analytic(q1, q2, t):
    
    v1_b = params.v1_bulk_background
    n_b  = params.density_background

    pert_real_v1 = 0
    pert_imag_v1 = 0
    pert_v1      = pert_real_v1 + 1j * pert_imag_v1
                   
    v1_analytic = v1_b + params.amplitude * pert_v1 * \
                    np.exp(  1j * (params.k_q1 * q1 + params.k_q2 * q2) 
                           + omega * t
                          ).real
    return(v1_analytic)

def v2_analytic(q1, q2, t):
    
    v2_b = params.v2_bulk_background
    n_b  = params.density_background

    pert_real_v2 = 0
    pert_imag_v2 = 0
    pert_v2      = pert_real_v2 + 1j * pert_imag_v2
                   
    v2_analytic = v2_b + params.amplitude * pert_v2 * \
                    np.exp(  1j * (params.k_q1 * q1 + params.k_q2 * q2) 
                           + omega * t
                          ).real
    
    return(v2_analytic)

def T_analytic(q1, q2, t):
    
    T_b = params.temperature_background
    n_b = params.density_background

    pert_real_T = -T_b / n_b
    pert_imag_T = 0
    pert_T      = pert_real_T + 1j * pert_imag_T

    T_analytic = T_b + params.amplitude * pert_T * \
                  np.exp(  1j * (params.k_q1 * q1 + params.k_q2 * q2) 
                         + omega * t
                        ).real

    return(T_analytic)

N_g_q = domain.N_ghost_q
N     = np.array([32, 48, 64, 96, 128])

error_n  = np.zeros(N.size)
error_v1 = np.zeros(N.size)
error_v2 = np.zeros(N.size)
error_T  = np.zeros(N.size)

for i in range(N.size):

    dq1 = (domain.q1_end - domain.q1_start) / int(N[i])
    dq2 = (domain.q2_end - domain.q2_start) / int(N[i])
    
    q1 = domain.q1_start + (0.5 + np.arange(int(N[i]))) * dq1
    q2 = domain.q2_start + (0.5 + np.arange(int(N[i]))) * dq2

    q2, q1 = np.meshgrid(q2, q1)

    h5f = h5py.File('dump/N_%04d'%(int(N[i])) + '.h5')
    mom = h5f['moments'][:]
    h5f.close()

    n_nls  = np.transpose(mom[:, :, 0], (1, 0))
    v1_nls = np.transpose(mom[:, :, 2], (1, 0)) / n_nls
    v2_nls = np.transpose(mom[:, :, 3], (1, 0)) / n_nls
    v3_nls = np.transpose(mom[:, :, 4], (1, 0)) / n_nls
    T_nls  = (1 / params.p_dim) * (  2 * np.transpose(mom[:, :, 1], (1, 0))
                                   - n_nls * v1_nls**2
                                   - n_nls * v2_nls**2
                                   - n_nls * v3_nls**2
                                  ) / n_nls

    n_analytic  = n_analytic(q1, q2, params.t_final)
    v1_analytic = v1_analytic(q1, q2, params.t_final)
    v2_analytic = v2_analytic(q1, q2, params.t_final)
    T_analytic  = T_analytic(q1, q2, params.t_final)

    error_n[i]  = np.mean(abs(n_nls - n_analytic))
    error_v1[i] = np.mean(abs(v1_nls - v1_analytic))
    error_v2[i] = np.mean(abs(v2_nls - v2_analytic))
    error_T[i]  = np.mean(abs(T_nls - T_analytic))

print('Errors Obtained:')
print('L1 norm of error for density:', error_n)
print('L1 norm of error for velocity-1:', error_v1)
print('L1 norm of error for velocity-2:', error_v2)
print('L1 norm of error for temperature:', error_T)

print('\nConvergence Rates:')
print('Order of convergence for density:', np.polyfit(np.log10(N), np.log10(error_n), 1)[0])
print('Order of convergence for velocity-1:', np.polyfit(np.log10(N), np.log10(error_v1), 1)[0])
print('Order of convergence for velocity-2:', np.polyfit(np.log10(N), np.log10(error_v2), 1)[0])
print('Order of convergence for temperature:', np.polyfit(np.log10(N), np.log10(error_T), 1)[0])

pl.loglog(N, error_n, '-o', label = 'Density')
pl.loglog(N, error_v1, '-o', label = 'Velocity-1')
pl.loglog(N, error_v2, '-o', label = 'Velocity-2')
pl.loglog(N, error_T, '-o', label = 'Temperature')
pl.loglog(N, error_n[0]*32**2/N**2, '--', color = 'black', label = r'$O(N^{-2})$')
pl.xlabel(r'$N$')
pl.ylabel('Error')
pl.legend()
pl.savefig('convergenceplot.png')
