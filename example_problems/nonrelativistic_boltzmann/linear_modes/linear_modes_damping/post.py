import numpy as np
import h5py
import matplotlib as mpl
mpl.use('agg')
import pylab as pl

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

N = np.array([64, 96, 128, 192, 256, 384, 512])

for i in range(N.size):
    
    h5f        = h5py.File('dump/data_Nx_'+ str(int(N[i])) + '_Nv_' + str(int(N[i])) + '.h5', 'r')
    n_nls      = h5f['n_nls'][:]
    n_nls2     = h5f['n_nls2'][:]
    time_array = h5f['time'][:]
    h5f.close()

    h5f         = h5py.File('dump/data_Nx_64_Nv_1024.h5', 'r')
    n_ls        = h5f['n_ls'][:]
    time_array2 = h5f['time'][:]
    h5f.close()

    pl.plot(time_array, n_nls2, label = 'Without Filter')
    pl.plot(time_array, n_nls, '--', color = 'red', label = 'With Filter')
    pl.plot(time_array2, n_ls, '--', color = 'black', label = 'Linear Theory')
    pl.ylabel('$n$')
    pl.xlabel('Time')
    pl.legend(fontsize = 25)
    pl.savefig('plots/n' + str(int(N[i])) + '.png', bbox_inches = 'tight')
    pl.clf()

    pl.semilogy(time_array, n_nls2-1, label = 'Without Filter')
    pl.semilogy(time_array, n_nls-1, '--', color = 'red', label = 'With Filter')
    pl.plot(time_array2, n_ls-1, '--', color = 'black', label = 'Linear Theory')
    pl.ylabel(r'$\delta n$')
    pl.xlabel('Time')
    pl.legend(fontsize = 25)
    pl.savefig('plots/n_semilogy_' + str(int(N[i])) + '.png', bbox_inches = 'tight')
    pl.clf()
