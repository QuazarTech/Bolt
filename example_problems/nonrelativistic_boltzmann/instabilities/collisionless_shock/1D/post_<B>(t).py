import numpy as np
import matplotlib
matplotlib.use('agg')
import pylab as pl

import h5py
import domain
import params

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 9, 4
pl.rcParams['figure.dpi']      = 300
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

time_array = np.arange(0, 1000 * params.t0 + params.dt_dump_moments, 
                       params.dt_dump_moments
                      )

B_mean_data = np.zeros(time_array.size)

for time_index, t0 in enumerate(time_array):
    
    h5f    = h5py.File('dump_fields/t=%.3f'%(t0) + '.h5', 'r')
    fields = np.swapaxes(h5f['EM_fields'][:], 0, 1)
    h5f.close()

    B1 = fields[0, :, 3] / params.B0
    B2 = fields[0, :, 4] / params.B0
    B3 = fields[0, :, 5] / params.B0
    
    # if(time_index % 1000 == 0):
    #     pl.plot(B1 / params.B0)
    #     pl.show()

    B_mean_data[time_index] = np.mean(B1**2 + B2**2 + B3**2)

# h5f = h5py.File('B_mean_data.h5', 'w')
# h5f.create_dataset('B_mean', data = B_mean_data)
# h5f.close()

# h5f = h5py.File('B_mean_data.h5', 'r')
# B_mean_data = h5f['B_mean'][:] / params.B0
# h5f.close()

pl.semilogy(time_array / params.t0, B_mean_data, label = r'$<B_x^2>$')
pl.xlabel(r'Time($\omega_c^{-1})$')
pl.ylabel(r'$<B^2>$')
pl.xlim([0, 200])
# pl.legend()
pl.savefig('plot.png', bbox_inches = 'tight')
