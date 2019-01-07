import numpy as np
import matplotlib
import pylab as pl

import h5py
import domain
import params

from post import return_array_to_be_plotted, return_field_to_be_plotted
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

N_q1 = domain.N_q1
N_q2 = domain.N_q2
N_g  = domain.N_ghost

dq1 = (domain.q1_end - domain.q1_start) / N_q1
dq2 = (domain.q2_end - domain.q2_start) / N_q2

time_array = np.arange(0, params.t_final + params.dt_dump_moments, 
                       params.dt_dump_moments
                      )

JdotE = np.zeros(time_array.size)

for time_index, t0 in enumerate(time_array):

    h5f     = h5py.File('dump_moments/t=%.3f'%(t0) + '.h5', 'r')
    moments = np.swapaxes(h5f['moments'][:], 0, 1)
    h5f.close()

    h5f    = h5py.File('dump_fields/t=%.3f'%(t0) + '.h5', 'r')
    fields = np.swapaxes(h5f['EM_fields'][:], 0, 1)
    h5f.close()

    # This is evaluated at 
    J1 = return_array_to_be_plotted('J1', moments)
    J2 = return_array_to_be_plotted('J2', moments)
    J3 = return_array_to_be_plotted('J3', moments)

    E1 = return_field_to_be_plotted('E1', fields) # (i)
    # Getting E1 at (i + 1/2):
    E1 = 0.5 * (np.roll(E1, -1) + E1) # (i + 1/2)
    E2 = return_field_to_be_plotted('E2', fields) # (i + 1/2)
    E3 = return_field_to_be_plotted('E3', fields) # (i + 1/2)

    JdotE[time_index] = np.sum((J1 * E1 + J2 * E2 + J3 * E3)) * dq1

pl.semilogy(time_array / params.t0, abs(JdotE))
pl.xlabel(r'Time($\omega_c^{-1})$')
pl.ylabel(r'$|<\mathbf{J}\cdot\mathbf{E}>|$')
# pl.yscale('symlog')
# pl.legend()
pl.savefig('plot.png', bbox_inches = 'tight')
