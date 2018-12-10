# File used to check if the constraints are preserved throughout:

import numpy as np
import matplotlib
matplotlib.use('agg')
import pylab as pl

import h5py
import domain
import params

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 15, 10
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

dq1 = (domain.q1_end - domain.q1_start) / domain.N_q1
dq2 = (domain.q2_end - domain.q2_start) / domain.N_q2

# Array used to hold the value of MEAN(|divE - rho|):
gauss_law_electric = np.zeros(time_array.size)
# Array used to hold the value of MEAN(|divB|):
gauss_law_magnetic = np.zeros(time_array.size)

for time_index, t0 in enumerate(time_array):
    
    h5f  = h5py.File('dump_moments/t=%.3f'%(t0) + '.h5', 'r')
    moments = np.swapaxes(h5f['moments'][:], 0, 1)
    h5f.close()

    h5f    = h5py.File('dump_fields/t=%.3f'%(t0) + '.h5', 'r')
    fields = np.swapaxes(h5f['EM_fields'][:], 0, 1)
    h5f.close()

    E1 = fields[:, :, 0]
    E2 = fields[:, :, 1]
    E3 = fields[:, :, 2]
    B1 = fields[:, :, 3]
    B2 = fields[:, :, 4]
    B3 = fields[:, :, 5]

    n          = moments[:, :, 0:2]
    n[:, :, 0] = -1 * n[:, :, 0]
    n          = np.sum(n, 2)

    divB = (B1 -  np.roll(B1, 1, 0)) / dq1 + (B2  - np.roll(B2, 1, 1)) / dq2
    divE = (np.roll(E1, -1, 0) - E1) / dq1 + (np.roll(E2, -1, 1) - E2) / dq2

    gauss_law_magnetic[time_index] = np.mean(abs(divB))
    gauss_law_electric[time_index] = np.mean(abs(divE - n / params.eps))

print(gauss_law_magnetic)
print(gauss_law_electric)

pl.semilogy(time_array, gauss_law_magnetic, label = r'$|\nabla \cdot \vec{E} - \rho|$')
pl.semilogy(time_array, gauss_law_electric, label = r'$|\frac{d \rho}{d t} + \nabla \cdot \vec{J}|$')
pl.xlabel('Time')
pl.ylabel()
pl.legend()
pl.savefig('plot.png')
