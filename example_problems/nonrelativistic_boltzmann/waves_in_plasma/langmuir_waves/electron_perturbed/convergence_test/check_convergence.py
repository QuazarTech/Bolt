import numpy as np
import h5py
import matplotlib as mpl 
mpl.use('agg')
import pylab as pl

from .. import domain

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

N = 2**np.arange(5, 11)

error_n = np.zeros(N.size)

for i in range(N.size):

    h5f   = h5py.File('dump/ls_N_%04d'%(int(N[i])) + '.h5', 'r')
    f_hat = h5f['f_hat'][:]
    h5f.close()

    h5f   = h5py.File('dump/nls_N_%04d'%(int(N[i])) + '.h5', 'r')
    f_nls = h5f['distribution_function'][:]
    h5f.close()

    f_background   = abs(f_hat)[:, 0, 0]
    f_hat[:, 0, 0] = 0

    delta_f_hat = f_hat[:, af.imax(abs(np.sum(np.sum(f_hat, 2), 0)))[1], 0]

    dq1 = (domain.q1_end - domain.q1_start) / domain.N_q1
    q1  = domain.q1_start + (0.5 + np.arange(domain.N_q1)) * dq1
    q1  = np.tile(q1.reshape(1, 1, domain.N_q1), (1, 3, 1))

    f_ls       = f_background + (delta_f_hat * np.exp(-1j * 4 * np.pi * q1)).real
    f_ls       = np.transpose(f_ls, (1, 2, 0))
    error_n[i] = np.mean(abs(f_nls - f_ls))

print('Error Obtained:')
print('L1 norm of error:', error_n)

print('\nConvergence Rate:')
print('Order of convergence:', np.polyfit(np.log10(N), np.log10(error), 1)[0])

pl.loglog(N, error, '-o', label = 'Numerical')
pl.loglog(N, error[0]*32**2/N**2, '--', color = 'black', label = r'$O(N^{-2})$')
pl.xlabel(r'$N$')
pl.ylabel('Error')
pl.legend()
pl.savefig('convergenceplot.png')
