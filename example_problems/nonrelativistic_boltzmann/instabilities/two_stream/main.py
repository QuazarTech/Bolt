import arrayfire as af
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import pylab as pl
import h5py

from bolt.lib.physical_system import physical_system
from bolt.lib.nonlinear.nonlinear_solver import nonlinear_solver
from bolt.lib.linear.linear_solver import linear_solver
from bolt.lib.utils.fft_funcs import ifft2

import domain
import boundary_conditions
import initialize
import params

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

# Declaring a linear system object which will evolve the defined physical system:
nls = nonlinear_solver(system)
N_g = nls.N_ghost
ls  = linear_solver(system)

# Time parameters:
dt = params.N_cfl * min(nls.dq1, nls.dq2) \
                  / max(domain.p1_end + domain.p2_end + domain.p3_end)

time_array = np.arange(0, params.t_final + dt, dt)

# Initializing Arrays used in storing the data:
E_data_ls  = np.zeros_like(time_array)
E_data_nls = np.zeros_like(time_array)

ke_data_ls  = np.zeros_like(time_array)
ke_data_nls = np.zeros_like(time_array)

for time_index, t0 in enumerate(time_array):
    
    if(time_index%10 == 0):
        nls.dump_distribution_function('dump_f/%04d'%time_index)

    print('Computing For Time =', t0)

    E_data_nls[time_index] = af.max(nls.fields_solver.cell_centered_EM_fields[:, :, N_g:-N_g, N_g:-N_g])
    E1_ls                  = af.real(0.5 * (ls.N_q1 * ls.N_q2) 
                                         * ifft2(ls.fields_solver.E1_hat)
                                    )

    E_data_ls[time_index]  = af.max(E1_ls)

    ke_data_nls[time_index] = af.mean(nls.compute_moments('mom_v1_bulk')**2 / nls.compute_moments('density'))
    ke_data_ls[time_index]  = af.mean(ls.compute_moments('mom_v1_bulk')**2 / ls.compute_moments('density'))

    nls.strang_timestep(dt)
    # ls.RK4_timestep(dt)

import h5py
h5f = h5py.File('data.h5', 'w')
h5f.create_dataset('electrical_field_ls', data = E_data_ls)
h5f.create_dataset('electrical_field_nls', data = E_data_nls)
h5f.create_dataset('kinetic_energy_ls', data = ke_data_ls)
h5f.create_dataset('kinetic_energy_nls', data = ke_data_nls)
h5f.create_dataset('time', data = time_array)
h5f.close()

pl.plot(time_array, E_data_nls, label='Nonlinear Solver')
#pl.plot(time_array, E_data_ls, '--', color = 'black', label = 'Linear Solver')
pl.ylabel(r'MAX($|E|$)')
pl.xlabel('Time')
#pl.legend()
pl.savefig('linearplot.png')
pl.clf()

pl.semilogy(time_array, E_data_nls, label='Nonlinear Solver')
#pl.semilogy(time_array, E_data_ls, '--', color = 'black', label = 'Linear Solver')
pl.ylabel(r'MAX($|E|$)')
pl.xlabel('Time')
#pl.legend()
pl.savefig('semilogyplot.png')
pl.clf()

pl.plot(time_array, ke_data_nls, label='Nonlinear Solver')
#pl.plot(time_array, ke_data_ls, '--', color = 'black', label = 'Linear Solver')
pl.ylim(-0.02, 0.02)
pl.ylabel(r'MEAN(KE)')
pl.xlabel('Time')
#pl.legend()
pl.savefig('ke_linearplot.png')
pl.clf()

pl.semilogy(time_array, ke_data_nls, label='Nonlinear Solver')
#pl.semilogy(time_array, ke_data_ls, '--', color = 'black', label = 'Linear Solver')
pl.ylabel(r'MEAN(KE)')
pl.xlabel('Time')
#pl.legend()
pl.savefig('ke_semilogyplot.png')
pl.clf()
