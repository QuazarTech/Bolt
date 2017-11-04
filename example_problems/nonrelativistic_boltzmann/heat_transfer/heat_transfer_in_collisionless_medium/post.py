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

# Analytical:
h5f = h5py.File('analytical.h5', 'r')
temp_ana = h5f['temperature'][:]
time_ana = h5f['time'][:]
h5f.close()

# Numerical:
h5f = h5py.File('numerical.h5', 'r')
temp_num = h5f['temperature'][:]
time_num = h5f['time'][:]
h5f.close()

# Plotting:
pl.plot(time_num, temp_num, label = 'Numerical')
pl.plot(time_ana, temp_ana, '--', color = 'black', label = 'Analytical')
pl.xlabel(r'$t$')
pl.ylabel(r'$T_{avg}$')
pl.legend()
pl.savefig('plot.png')
