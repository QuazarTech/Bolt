import numpy as np
import h5py
import matplotlib as mpl
mpl.use('agg')
import pylab as pl

import domain
import params

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 9, 4
pl.rcParams['figure.dpi']      = 300
pl.rcParams['image.cmap']      = 'bwr'
pl.rcParams['lines.linewidth'] = 1.5
pl.rcParams['font.family']     = 'serif'
pl.rcParams['font.weight']     = 'bold'
pl.rcParams['font.size']       = 30
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

h5f        = h5py.File('data.h5', 'r')
n_nls      = h5f['n_nls'][:]
E_nls      = h5f['E_nls'][:]
time_array = h5f['time'][:]
h5f.close()

h5f         = h5py.File('data_ls.h5', 'r')
n_ls        = h5f['n_ls'][:]
E_ls        = h5f['E_ls'][:]
time_array2 = h5f['time'][:]
h5f.close()

# pl.plot(time_array, n_nls, label = r'{\tt bolt}')
# pl.plot(time_array, n_ls, '--', color = 'black', label = 'Linear Theory')
# pl.ylabel(r'$n$')
# pl.xlabel(r'Time (${\omega_p}^{-1}$)')
# pl.legend()
# pl.savefig('n.png')
# pl.clf()

# pl.plot(time_array, E_nls, label = r'{\tt bolt}')
# #pl.plot(time_array, E_ls, '--', color = 'black', label = 'Linear Theory')
# pl.ylabel(r'MAX($E$)')
# pl.xlabel(r'Time (${\omega_p}^{-1}$)')
# #pl.legend()
# pl.savefig('E.png')
# pl.clf()

# pl.semilogy(time_array, n_nls, label = r'{\tt bolt}')
# pl.semilogy(time_array, n_ls, '--', color = 'black', label = 'Linear Theory')
# pl.ylabel(r'$n$')
# pl.xlabel(r'Time (${\omega_p}^{-1}$)')
# pl.legend()
# pl.savefig('n_semilogy.png')
# pl.clf()

pl.semilogy(time_array, E_nls, label = r'{\tt bolt}')
# pl.semilogy(time_array2, E_ls, '--', color = 'black', label = 'Linear Theory')
pl.ylabel(r'MAX($E$)')
pl.xlabel(r'Time (${\omega_p}^{-1}$)')
# pl.legend()
pl.savefig('E_semilogy.png', bbox_inches = 'tight')
pl.clf()
