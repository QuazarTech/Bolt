import arrayfire as af
import numpy as np
import pylab as pl

from bolt.lib.physical_system import physical_system
from bolt.lib.nonlinear.nonlinear_solver import nonlinear_solver
# from bolt.lib.linear.linear_solver import linear_solver

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

# Pass this system to the linear solver object when 
# a single mode only needs to be evolved. This solver
# would only evolve the single mode, and hence requires
# much lower time and memory for the computations:
# linearized_system = physical_system(domain,
#                                     boundary_conditions,
#                                     params,
#                                     initialize,
#                                     advection_terms,
#                                     collision_operator.linearized_BGK,
#                                     moments
#                                    )

# Declaring a linear system object which will evolve the defined physical system:
nls = nonlinear_solver(system)
# ls  = linear_solver(system)

# Timestep as set by the CFL condition:
dt = params.N_cfl * min(nls.dq1, nls.dq2) \
                  / max(domain.p1_end, domain.p2_end, domain.p3_end)

time_array  = np.arange(0, params.t_final + dt, dt)

rho_data_nls = np.zeros(time_array.size)
# rho_data_ls  = np.zeros(time_array.size)

# Storing data at time t = 0:
n_nls           = nls.compute_moments('density')
rho_data_nls[0] = af.max(n_nls[:, :, N_g_q:-N_g_q, N_g_q:-N_g_q])

# n_ls = ls.compute_moments('density')

# if(ls.single_mode_evolution == True):
#     rho_data_ls[0] = params.rho_background + abs(n_ls)
# else:
#     rho_data_ls[0] = af.max(n_ls) 

for time_index, t0 in enumerate(time_array[1:]):

    nls.strang_timestep(dt)
    # ls.RK4_timestep(dt)

    n_nls                         = nls.compute_moments('density')
    rho_data_nls[time_index + 1]  = af.max(n_nls[:, 0, N_g_q:-N_g_q, N_g_q:-N_g_q])
    
    # n_ls = ls.compute_moments('density')

    # if(ls.single_mode_evolution == True):
    #     rho_data_ls[time_index + 1] =  params.rho_background + abs(n_ls)
    # else:
    #     rho_data_ls[time_index + 1] = af.max(n_ls) 

# pl.plot(time_array, rho_data_ls, '--', color = 'black', label = 'Linear Solver')
pl.plot(time_array, rho_data_nls, label='Nonlinear Solver')
pl.ylabel(r'MAX($\rho$)')
pl.xlabel('Time')
pl.legend()
pl.savefig('rho.png')
pl.clf()
