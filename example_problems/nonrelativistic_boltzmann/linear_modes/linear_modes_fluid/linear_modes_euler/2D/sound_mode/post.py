import numpy as np
import h5py
import matplotlib as mpl 
mpl.use('agg')
import pylab as pl

import input_files.domain as domain
import input_files.params as params

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 20, 10
pl.rcParams['figure.dpi']      = 300
pl.rcParams['image.cmap']      = 'gist_heat'
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

omega =   np.sqrt(params.temperature_background * params.gamma) \
        * np.sqrt(params.k_q1**2 + params.k_q2**2) * 1j

# Defining the functions for the analytical solution:
def n_analytic(q1, q2, t):
    
    n_b = params.density_background

    pert_real_n = 1
    pert_imag_n = 0
    pert_n      = pert_real_n + 1j * pert_imag_n

    n_analytic = n_b + params.amplitude * pert_n * \
                  np.exp(  1j * (params.k_q1 * q1 + params.k_q2 * q2)  
                         + omega * t
                        ).real

    return(n_analytic)

dq1 = (domain.q1_end - domain.q1_start) / domain.N_q1
dq2 = (domain.q2_end - domain.q2_start) / domain.N_q2

# Timestep as set by the CFL condition:
dt = params.N_cfl * min(dq1, dq2) \
                  / max(domain.p1_end, domain.p2_end, domain.p3_end)

time_array = np.arange(0, params.t_final + dt, dt)

# Checking that time array doesn't cross final time:
if(time_array[-1]>params.t_final):
    time_array = np.delete(time_array, -1)

q1 = domain.q1_start + (0.5 + np.arange(domain.N_q1)) * dq1
q2 = domain.q2_start + (0.5 + np.arange(domain.N_q2)) * dq2

q2, q1 = np.meshgrid(q2, q1)

for time_index, t0 in enumerate(time_array):
    
    h5f   = h5py.File('dump/%04d'%time_index + '.h5')
    n_nls = np.transpose(h5f['moments'][:][:, :, 0], (1, 0))
    h5f.close()

    n_analytic = n_analytic(q1, q2, t0)

    fig = pl.figure()

    ax1 = fig.add_subplot(1,2,1)
    ax1.set_aspect('equal')
    ax2.contourf(q1, q2, n_analytic, 100)

    ax2 = fig.add_subplot(1,2,2)
    ax2.set_aspect('equal')
    ax2.contourf(q1, q2, n_nls, 100)

    fig.suptitle('Time = %.3f'%(t0))
    pl.savefig('images/' + '%04d'%time_index + '.png')
    pl.close(fig)
    pl.clf()
