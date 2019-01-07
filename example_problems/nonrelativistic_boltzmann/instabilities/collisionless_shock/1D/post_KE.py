import numpy as np
import matplotlib
import pylab as pl

import h5py
import domain
import params

from post import return_moment_to_be_plotted
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

KE_electrons = np.zeros(time_array.size)
KE_ions      = np.zeros(time_array.size)

for time_index, t0 in enumerate(time_array):
    
    h5f     = h5py.File('dump_moments/t=%.3f'%(t0) + '.h5', 'r')
    # dump_moments writes files in the structure (q2, q1, N_s)
    # But the post-processing functions require it in the form (q1, q2, N_s)
    # By using swapaxes we change (q2, q1, N_s) --> (q1, q2, N_s)
    moments = np.swapaxes(h5f['moments'][:], 0, 1)
    h5f.close()

    E  = return_moment_to_be_plotted('energy', moments)
    Ee = E[:, :, 0] / params.v0**2
    Ei = E[:, :, 1] / params.v0**2

    KE_electrons[time_index] = np.sum(Ee) * dq1 * dq2
    KE_ions[time_index]      = np.sum(Ei) * dq1 * dq2

pl.semilogy(time_array / params.t0, KE_electrons, label = r'Electrons')
pl.semilogy(time_array / params.t0, KE_ions, '--', color = 'C3', label = r'Ions')
pl.xlabel(r'Time($\omega_c^{-1})$')
pl.ylabel(r'Kinetic Energy')
pl.legend()
pl.savefig('plot.png', bbox_inches = 'tight')
