import numpy as np
import matplotlib
matplotlib.use('agg')
import pylab as pl

import h5py
import domain
import params

from post import return_field_to_be_plotted

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

time_array = np.arange(0, params.t_final + params.dt_dump_moments, 
                       params.dt_dump_moments
                      )

B_mean_data = np.zeros(time_array.size)

for time_index, t0 in enumerate(time_array):
    
    h5f    = h5py.File('dump_fields/t=%.3f'%(t0) + '.h5', 'r')
    # dump_EM_fields writes files in the structure (q2, q1, N_s)
    # But the post-processing functions require it in the form (q1, q2, N_s)
    # By using swapaxes we change (q2, q1, N_s) --> (q1, q2, N_s)
    fields = np.swapaxes(h5f['EM_fields'][:], 0, 1)
    h5f.close()

    B1 = return_field_to_be_plotted('B1', fields) / params.B0
    B2 = return_field_to_be_plotted('B2', fields) / params.B0
    B3 = return_field_to_be_plotted('B3', fields) / params.B0

    B_mean_data[time_index] = np.mean(B1**2 + B2**2 + B3**2)

pl.semilogy(time_array * params.plasma_frequency, B_mean_data)
pl.xlabel(r'Time($\omega_p^{-1})$')
pl.ylabel(r'$<B^2>$')
pl.xlim([0, 1000])
pl.savefig('plot.png', bbox_inches = 'tight')
