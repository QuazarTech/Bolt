import arrayfire as af
import numpy as np
import pylab as pl
import petsc4py
import sys
petsc4py.init(sys.argv)

from bolt.lib.physical_system import physical_system

from bolt.lib.nonlinear_solver.nonlinear_solver \
    import nonlinear_solver

from bolt.lib.linear_solver.linear_solver import linear_solver

import domain
import boundary_conditions
import params
import initialize

import bolt.src.nonrelativistic_boltzmann.advection_terms as advection_terms

import bolt.src.nonrelativistic_boltzmann.collision_operator \
    as collision_operator

import bolt.src.nonrelativistic_boltzmann.moment_defs as moment_defs

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
                         moment_defs
                        )

# Declaring a linear system object which will evolve the defined physical system:
nls = nonlinear_solver(system)
ls  = linear_solver(system)

# Time parameters:
dt      = 0.001
t_final = 0.5

time_array = np.arange(0, t_final + dt, dt)

# Initializing Arrays used in storing the data:
rho_data_nls  = np.zeros_like(time_array)

p1b_data_nls  = np.zeros_like(time_array)
p2b_data_nls  = np.zeros_like(time_array)
p3b_data_nls  = np.zeros_like(time_array)

temp_data_nls = np.zeros_like(time_array)

rho_data_ls  = np.zeros_like(time_array)

p1b_data_ls  = np.zeros_like(time_array)
p2b_data_ls  = np.zeros_like(time_array)
p3b_data_ls  = np.zeros_like(time_array)

temp_data_ls = np.zeros_like(time_array)

def time_evolution():

    for time_index, t0 in enumerate(time_array):
        print('Computing For Time =', t0)

        n_nls = nls.compute_moments('density')

        p1_bulk_nls = nls.compute_moments('mom_p1_bulk') / n_nls
        p2_bulk_nls = nls.compute_moments('mom_p2_bulk') / n_nls
        p3_bulk_nls = nls.compute_moments('mom_p3_bulk') / n_nls

        E_nls = nls.compute_moments('energy')

        T_nls = (  nls.compute_moments('energy')
                 - n_nls * p1_bulk_nls**2
                 - n_nls * p2_bulk_nls**2
                 - n_nls * p3_bulk_nls**2
                ) / n_nls

        rho_data_nls[time_index]  = af.max(n_nls)
        
        p1b_data_nls[time_index]  = af.max(p1_bulk_nls)
        p2b_data_nls[time_index]  = af.max(p2_bulk_nls)
        p3b_data_nls[time_index]  = af.max(p3_bulk_nls)

        temp_data_nls[time_index] = af.max(T_nls)

        n_ls = ls.compute_moments('density')

        p1_bulk_ls = ls.compute_moments('mom_p1_bulk') / n_ls
        p2_bulk_ls = ls.compute_moments('mom_p2_bulk') / n_ls
        p3_bulk_ls = ls.compute_moments('mom_p3_bulk') / n_ls

        T_ls = (  ls.compute_moments('energy')
                - n_ls * p1_bulk_ls**2
                - n_ls * p2_bulk_ls**2
                - n_ls * p3_bulk_ls**2
               ) / n_ls

        rho_data_ls[time_index]  = af.max(n_ls)
        
        p1b_data_ls[time_index]  = af.max(p1_bulk_ls)
        p2b_data_ls[time_index]  = af.max(p2_bulk_ls)
        p3b_data_ls[time_index]  = af.max(p3_bulk_ls)

        temp_data_ls[time_index] = af.max(T_ls)

        nls.strang_timestep(dt)
        ls.RK2_timestep(dt)
        
time_evolution()

pl.plot(time_array, rho_data_ls, '--', color = 'black', label = 'Linear Solver')
pl.plot(time_array, rho_data_nls, label='Nonlinear Solver')
pl.ylabel(r'MAX($\rho$)')
pl.xlabel('Time')
pl.legend()
pl.savefig('rho.png')
pl.clf()

pl.plot(time_array, temp_data_ls, '--', color = 'black', label = 'Linear Solver')
pl.plot(time_array, temp_data_nls, label='Nonlinear Solver')
pl.ylabel(r'MAX($T$)')
pl.xlabel('Time')
pl.legend()
pl.savefig('E.png')
pl.clf()

pl.plot(time_array, p1b_data_ls, '--', color = 'black', label = 'Linear Solver')
pl.plot(time_array, p1b_data_nls, label='Nonlinear Solver')
pl.ylabel(r'MAX($p_{1b}$)')
pl.xlabel('Time')
pl.legend()
pl.savefig('p1b.png')
pl.clf()

pl.plot(time_array, p2b_data_ls, '--', color = 'black', label = 'Linear Solver')
pl.plot(time_array, p2b_data_nls, label='Nonlinear Solver')
pl.ylabel(r'MAX($p_{2b}$)')
pl.xlabel('Time')
pl.legend()
pl.savefig('p2b.png')
pl.clf()

pl.plot(time_array, p3b_data_ls, '--', color = 'black', label = 'Linear Solver')
pl.plot(time_array, p3b_data_nls, label='Nonlinear Solver')
pl.ylabel(r'MAX($p_{3b}$)')
pl.xlabel('Time')
pl.legend()
pl.savefig('p3b.png')
pl.clf()
