import arrayfire as af
import numpy as np
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

# Defining the physical system to be solved:
system = physical_system(domain,
                         boundary_conditions,
                         params,
                         initialize,
                         advection_terms,
                         collision_operator.BGK,
                         moments
                        )

N_g = system.N_ghost

# Declaring a linear system object which will evolve the defined physical system:
nls = nonlinear_solver(system)
ls  = linear_solver(system)

# Timestep as set by the CFL condition:
dt = params.N_cfl * min(nls.dq1, nls.dq2) \
                  / max(domain.p1_end, domain.p2_end, domain.p3_end)

time_array  = np.arange(0, params.t_final + dt, dt)

rho_data_nls = np.zeros([time_array.size, 2])
rho_data_ls  = np.zeros([time_array.size, 2])

# Storing data at time t = 0:
n_nls              = nls.compute_moments('density')
rho_data_nls[0, 0] = af.max(n_nls[:, 0, N_g:-N_g, N_g:-N_g])
rho_data_nls[0, 1] = af.max(n_nls[:, 1, N_g:-N_g, N_g:-N_g])

n_ls              = ls.compute_moments('density')
rho_data_ls[0, 0] = af.max(n_ls[:, 0]) 
rho_data_ls[0, 1] = af.max(n_ls[:, 1]) 

q1 = np.array(nls.q1_center[0, 0, :, 0]).ravel()
p1 = np.array(nls.p1_center[:, 0, 0, 0]).ravel()

p1, q1 = np.meshgrid(p1, q1)

for time_index, t0 in enumerate(time_array[1:]):
    
    # if(time_index%10 == 0):
    #     pl.contourf(p1, q1, 
    #                 np.swapaxes(np.array(nls.f[:, 0, :, 0]).reshape(nls.N_p1, nls.N_q1 + 6), 0, 1), 
    #                 100
    #                )
    #     pl.xlabel(r'$v$')
    #     pl.ylabel(r'$x$')
    #     pl.title('Time = %.2f'%(t0-dt))
    #     pl.savefig('images/%04d'%(time_index/10) + '.png')
    #     pl.clf()

    print('Computing For Time =', t0)
    
    nls.strang_timestep(dt)
    ls.RK4_timestep(dt)

    n_nls                           = nls.compute_moments('density')
    rho_data_nls[time_index + 1, 0] = af.max(n_nls[:, 0, N_g:-N_g, N_g:-N_g])
    rho_data_nls[time_index + 1, 1] = af.max(n_nls[:, 1, N_g:-N_g, N_g:-N_g])
    
    n_ls                           = ls.compute_moments('density')
    rho_data_ls[time_index + 1, 0] = af.max(n_ls[:, 0]) 
    rho_data_ls[time_index + 1, 1] = af.max(n_ls[:, 1]) 

pl.plot(time_array, rho_data_nls[:, 0], '--', color = 'C3', label = 'Electrons')
pl.plot(time_array, rho_data_nls[:, 1], color = 'C0', label='Ions')
pl.ylabel(r'MAX($\rho$)')
pl.xlabel('Time')
pl.legend()
pl.savefig('rho_nonlinear.png')
pl.clf()

pl.plot(time_array, rho_data_ls[:, 0], '--', color = 'C3', label = 'Electrons')
pl.plot(time_array, rho_data_ls[:, 1], color = 'C0', label='Ions')
pl.ylabel(r'MAX($\rho$)')
pl.xlabel('Time')
pl.legend()
pl.savefig('rho_linear.png')
pl.clf()

pl.plot(time_array, rho_data_nls[:, 0], color = 'C3', label = 'Electrons(Nonlinear Solver)')
pl.plot(time_array, rho_data_nls[:, 1], color = 'C0', label = 'Ions(Nonlinear Solver)')
pl.plot(time_array, rho_data_ls[:, 0], '--', color = 'C3', label = 'Electrons(Linear Solver)')
pl.plot(time_array, rho_data_ls[:, 1], '--', color = 'C0', label = 'Ions(Linear Solver)')
pl.ylabel(r'MAX($\rho$)')
pl.xlabel('Time')
pl.legend()
pl.savefig('rho.png')
pl.clf()

pl.plot(time_array, rho_data_nls[:, 0], label = 'Nonlinear Solver')
pl.plot(time_array, rho_data_ls[:, 0], '--', color = 'black', label = 'Linear Solver')
pl.ylabel(r'MAX($\rho$)')
pl.xlabel('Time')
pl.legend()
pl.savefig('rho_e.png')
pl.clf()

pl.plot(time_array, rho_data_nls[:, 1], label = 'Nonlinear Solver')
pl.plot(time_array, rho_data_ls[:, 1], '--', color = 'black', label = 'Linear Solver')
pl.ylabel(r'MAX($\rho$)')
pl.xlabel('Time')
pl.legend()
pl.savefig('rho_i.png')
pl.clf()
