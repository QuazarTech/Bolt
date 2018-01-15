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
import initialize

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

# Defining the function for the analytical solution:
def rho_ana(q1, t):
    
    rho_b         = params.rho_background

    pert_real_rho = params.pert_rho.real
    pert_imag_rho = params.pert_rho.imag

    rho_ana = rho_b + (  pert_real_rho * af.cos(params.k_q1 * q1)
                       - pert_imag_rho * af.sin(params.k_q1 * q1)
                      ) * np.exp(1j * params.omega * t).real

    return(rho_ana)

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
rho_data_nls = np.zeros([time_array.size])
rho_data_ana = np.zeros([time_array.size])

# Storing data at time t = 0:
n_nls           = nls.compute_moments('density')
rho_data_nls[0] = af.max(n_nls[:, 0, N_g_q:-N_g_q, N_g_q:-N_g_q])
rho_data_ana[0] = af.max(rho_ana(nls.q1_center, 0)) 

for time_index, t0 in enumerate(time_array[1:]):
    print('Computing For Time =', t0)
    nls.strang_timestep(dt)

    n_nls                        = nls.compute_moments('density')
    rho_data_nls[time_index + 1] = af.max(n_nls[:, 0, N_g_q:-N_g_q, N_g_q:-N_g_q])
    rho_data_ana[time_index + 1] = af.max(rho_ana(nls.q1_center, t0))    

pl.plot(time_array, rho_data_nls, label='Nonlinear Solver')
pl.plot(time_array, rho_data_ana, label='Analytic')
pl.ylabel(r'MAX($\rho$)')
pl.xlabel('Time')
pl.legend()
pl.savefig('rho.png')
pl.clf()
