import numpy as np
import pylab as pl
import h5py

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

N_x   = np.array([32, 64, 128, 256, 512])
error = np.zeros(N_x.size)

for i in range(N_x.size):
    # Analytical:
    h5f = h5py.File('analytical_' + str(N_x[i]) + '.h5', 'r')
    temp_ana = h5f['temperature'][:]
    h5f.close()

    # Numerical:
    h5f = h5py.File('numerical_' + str(N_x[i]) + '.h5', 'r')
    temp_num = h5f['temperature'][:]
    h5f.close()

    x = (np.arange(N_x[i]) + 0.5) * 1/N_x[i]

    pl.plot(x, temp_ana, '--', color = 'black', label = 'Analytical')
    pl.plot(x, temp_num, label = 'Numerical')
    pl.title(r'$N_x=$' + str(N_x[i]))
    pl.xlabel(r'$x$')
    pl.ylabel(r'$T$')
    pl.legend()
    pl.savefig(str(N_x[i])+'.png')
    pl.clf()

    error[i] = np.mean(abs(temp_num - temp_ana))

# Plotting:
pl.loglog(N_x, error, label = 'Numerical')
pl.loglog(N_x, error[0] * 32/N_x, '--', color = 'black', label = r'$O(N^{-1})$')
pl.xlabel(r'$N$')
pl.ylabel('Error')
pl.legend()
pl.savefig('plot.png')
