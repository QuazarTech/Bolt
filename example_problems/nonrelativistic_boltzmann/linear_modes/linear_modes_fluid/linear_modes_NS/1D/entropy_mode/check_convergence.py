import numpy as np
import h5py
import matplotlib as mpl 
mpl.use('agg')
import pylab as pl

import input_files.domain as domain
import input_files.params as params
from input_files.solve_linear_modes import solve_linear_modes

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

eigval, eigvecs = solve_linear_modes(params)
omega           = eigval[1]
# Defining the functions for the analytical solution:
def n_analytic(q1, t):
    
    n_b    = params.density_background
    pert_n = eigvecs[0, 1]

    n_analytic = n_b + params.amplitude * pert_n * \
                  np.exp(  1j * params.k_q1 * q1 
                         + omega * t
                        ).real

    return(n_analytic)

def v1_analytic(q1, t):
    
    v1_b    = params.v1_bulk_background
    pert_v1 = eigvecs[1, 1]
                   
    v1_analytic = v1_b + params.amplitude * pert_v1 * \
                    np.exp(  1j * params.k_q1 * q1 
                           + omega * t
                          ).real
    return(v1_analytic)

def T_analytic(q1, t):
    
    T_b    = params.temperature_background
    pert_T = eigvecs[2, 1]

    T_analytic = T_b + params.amplitude * pert_T * \
                  np.exp(  1j * params.k_q1 * q1 
                         + omega * t
                        ).real

    return(T_analytic)

N = 2**np.arange(5, 10)

error_n  = np.zeros(N.size)
error_v1 = np.zeros(N.size)
error_T  = np.zeros(N.size)

for i in range(N.size):

    dq1 = (domain.q1_end - domain.q1_start) / int(N[i])
    q1  = domain.q1_start + (0.5 + np.arange(int(N[i]))) * dq1

    h5f = h5py.File('dump/N_%04d'%(int(N[i])) + '.h5')
    mom = h5f['moments'][:]
    h5f.close()

    n_nls  = mom[0, :, 0].ravel()
    v1_nls = mom[0, :, 2].ravel() / n_nls
    v2_nls = mom[0, :, 3].ravel() / n_nls
    v3_nls = mom[0, :, 4].ravel() / n_nls
    T_nls  = (1 / params.p_dim) * (  2 * mom[0, :, 1].ravel() 
                                   - n_nls * v1_nls**2
                                   - n_nls * v2_nls**2
                                   - n_nls * v3_nls**2
                                  ) / n_nls

    n_analytic  = n_analytic(q1, params.t_final)
    v1_analytic = v1_analytic(q1, params.t_final)
    T_analytic  = T_analytic(q1, params.t_final)

    error_n[i]  = np.mean(abs(n_nls - n_analytic))
    error_v1[i] = np.mean(abs(v1_nls - v1_analytic))
    error_T[i]  = np.mean(abs(T_nls - T_analytic))

print('Errors Obtained:')
print('L1 norm of error for density:', error_n)
print('L1 norm of error for velocity:', error_v1)
print('L1 norm of error for temperature:', error_T)

print('\nConvergence Rates:')
print('Order of convergence for density:', np.polyfit(np.log10(N), np.log10(error_n), 1)[0])
print('Order of convergence for velocity:', np.polyfit(np.log10(N), np.log10(error_v1), 1)[0])
print('Order of convergence for temperature:', np.polyfit(np.log10(N), np.log10(error_T), 1)[0])

pl.loglog(N, error_n, '-o', label = 'Density')
pl.loglog(N, error_v1, '-o', label = 'Velocity')
pl.loglog(N, error_T, '-o', label = 'Temperature')
pl.loglog(N, error_n[0]*32**2/N**2, '--', color = 'black', label = r'$O(N^{-2})$')
pl.xlabel(r'$N$')
pl.ylabel('Error')
pl.legend()
pl.savefig('convergenceplot.png')
