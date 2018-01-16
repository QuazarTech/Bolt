import arrayfire as af
import numpy as np
import matplotlib as mpl 
mpl.use('agg')
import pylab as pl

from bolt.lib.physical_system import physical_system
from bolt.lib.nonlinear.nonlinear_solver import nonlinear_solver
from bolt.lib.linear.linear_solver import linear_solver

import domain
import boundary_conditions
import params
import initialize_sound_mode as initialize
# import initialize_entropy_mode as initialize

import bolt.src.nonrelativistic_boltzmann.advection_terms as advection_terms
import bolt.src.nonrelativistic_boltzmann.collision_operator as collision_operator
import bolt.src.nonrelativistic_boltzmann.moments as moments

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

omega = np.sqrt(params.temperature_background * params.gamma) * params.k_q1 * 1j

# Defining the functions for the analytical solution:
def n_analytic(q1, t):
    
    n_b = params.density_background

    pert_real_n = 1
    pert_imag_n = 0
    pert_n      = pert_real_n + 1j * pert_imag_n

    n_ana= n_b + params.amplitude * pert_n * \
                 np.exp(  1j * params.k_q1 * q1 
                        + omega * t
                       ).real

    return(n_ana)

# Defining the physical system to be solved:
system = physical_system(domain,
                         boundary_conditions,
                         params,
                         initialize,
                         advection_terms,
                         collision_operator.BGK,
                         moments
                        )

N_g_q = system.N_ghost_q

nls = nonlinear_solver(system)

# Timestep as set by the CFL condition:
dt = params.N_cfl * min(nls.dq1, nls.dq2) \
                  / max(domain.p1_end, domain.p2_end, domain.p3_end)

time_array   = np.arange(0, params.t_final + dt, dt)

# Storing data at time t = 0:
n_nls = np.array(nls.compute_moments('density'))
n_ana = n_analytic(np.array(nls.q1_center), 0)

pl.plot(np.array(nls.q1_center).reshape(38, 9)[3:-3, 0], 
        n_nls.reshape(38, 9)[3:-3, 0], 
        label = 'Nonlinear Solver'
       )
pl.plot(np.array(nls.q1_center).reshape(38, 9)[3:-3, 0], 
        n_ana.reshape(38, 9)[3:-3, 0], '--', color = 'black',
        label = 'Analytical'
       )
pl.xlabel(r'$x$')
pl.ylabel(r'$n$')
pl.legend()
pl.title('Time = 0')
pl.savefig('images/0000.png')
pl.clf()

for time_index, t0 in enumerate(time_array[1:]):
    
    print('Computing For Time =', t0)
    nls.strang_timestep(dt)

    n_nls = np.array(nls.compute_moments('density'))
    n_ana = n_analytic(np.array(nls.q1_center), t0)

    pl.plot(np.array(nls.q1_center).reshape(38, 9)[3:-3, 0], 
            n_nls.reshape(38, 9)[3:-3, 0], 
            label = 'Nonlinear Solver'
           )
    pl.plot(np.array(nls.q1_center).reshape(38, 9)[3:-3, 0], 
            n_ana.reshape(38, 9)[3:-3, 0], '--', color = 'black',
            label = 'Analytical'
           )
    pl.xlabel(r'$x$')
    pl.ylabel(r'$n$')
    pl.legend()
    pl.title('Time =%.2f'%t0)
    pl.savefig('images/%04d'%(time_index + 1) + '.png')
    pl.clf()
