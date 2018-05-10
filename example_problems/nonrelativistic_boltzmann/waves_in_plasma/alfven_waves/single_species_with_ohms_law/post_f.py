import numpy as np
import matplotlib
matplotlib.use('agg')
import pylab as pl

import h5py
import domain
import params

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 12.5, 7 #10, 14
pl.rcParams['figure.dpi']      = 80
pl.rcParams['image.cmap']      = 'jet'
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

# Declaration of the time array:
time_array = np.arange(0, 0.199 * params.t0 + params.dt_dump_moments, 
                       params.dt_dump_moments
                      )

for time_index, t0 in enumerate(time_array):
    
    h5f  = h5py.File('dump_f/t=%.3f'%(t0) + '.h5', 'r')
    f = np.swapaxes(h5f['distribution_function'][:], 0, 1)
    h5f.close()

    f = f.reshape(32, 3, 2, 32, 32, 32)

    f_to_plot = f[:, 0, 0, :, 0, 0].reshape(32, 32)

    pl.contourf(f_to_plot, 100)
    pl.title('Time = %.4f'%(t0 / params.t0) + r' $\tau_A^{-1}$')
    pl.savefig('images/%04d'%time_index + '.png')
