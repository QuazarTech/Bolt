import numpy as np
import h5py
import matplotlib as mpl 
mpl.use('agg')
import pylab as pl

import input_files.domain as domain
import input_files.params as params
from input_files.solve_linear_modes import solve_linear_modes

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


eigval, eigvecs = solve_linear_modes(params)
omega           = eigval[1]
# Defining the functions for the analytical solution:
def n_analytic(q1, t):
    
    n_b    = params.density_background
    pert_n = eigvecs[0, 1]

    n_analytic = n_b + params.amplitude * pert_n * \
                 np.exp(  1j * params.k_q1 * q1 
                        + omega * t
                       ).real

    return(n_analytic)

dq1 = (domain.q1_end - domain.q1_start) / domain.N_q1

# Timestep as set by the CFL condition:
dt = params.N_cfl * dq1 \
                  / max(domain.p1_end, domain.p2_end, domain.p3_end)

time_array = np.arange(0, params.t_final + dt, dt)

# Checking that time array doesn't cross final time:
if(time_array[-1]>params.t_final):
    time_array = np.delete(time_array, -1)

q1 = domain.q1_start + (0.5 + np.arange(domain.N_q1)) * dq1

for time_index, t0 in enumerate(time_array):
    
    h5f   = h5py.File('dump/%04d'%time_index + '.h5')
    n_nls = h5f['moments'][:][0, :, 0].ravel()
    h5f.close()

    n_analytic = n_analytic(q1, t0)

    pl.plot(q1, n_nls, 
            label = 'Nonlinear Solver'
           )
    pl.plot(q1, n_analytic, '--', color = 'black', 
            label = 'Analytical'
           )

    pl.xlabel(r'$x$')
    pl.ylabel(r'$n$')
    pl.legend()
    pl.title('Time =%.2f'%t0)
    pl.savefig('images/%04d'%time_index + '.png')
    pl.clf()
