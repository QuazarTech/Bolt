import arrayfire as af
import numpy as np
import h5py
import matplotlib as mpl
mpl.use('agg')
import pylab as pl

from bolt.lib.physical_system import physical_system
from bolt.lib.nonlinear.nonlinear_solver import nonlinear_solver

import domain
import boundary_conditions
import params
import initialize

import bolt.src.nonrelativistic_boltzmann.advection_terms as advection_terms
import bolt.src.nonrelativistic_boltzmann.collision_operator as collision_operator
import bolt.src.nonrelativistic_boltzmann.moments as moments

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 12, 7.5
pl.rcParams['figure.dpi']      = 100
pl.rcParams['image.cmap']      = 'bwr_r'
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

# Timestep as set by the CFL condition:
dt = params.N_cfl * min(nls.dq1, nls.dq2) \
                  / max(domain.p1_end, domain.p2_end, domain.p3_end)

time_array = np.arange(0, params.t_final + dt, dt)
# Array used to hold the value of MEAN(|divE - rho|)
gauss_law  = np.zeros(time_array.size)
# Array used to hold the value of MEAN(|d(rho)/dt + divJ|)
continuity = np.zeros(time_array.size)

continuity[0] = 0

# Gauss' Law:
rho       = -1 * nls.compute_moments('density')
rho[0, 1] = -1 * rho[0, 1]
rho       = af.sum(rho, 1)

# divE = nls.fields_solver.compute_divE()
# gauss_law[0] = af.mean(af.abs(divE - rho))

for time_index, t0 in enumerate(time_array[1:]):

    # Applying periodic boundary conditions:
    nls._communicate_f()

    rho_n       = -1 * nls.compute_moments('density')
    rho_n[0, 1] = -1 * rho_n[0, 1]

    pl.contourf(np.array(nls.q1_center[:, :, 3:-3, 3:-3]).reshape(256, 256), 
                np.array(nls.q2_center[:, :, 3:-3, 3:-3]).reshape(256, 256), 
                np.array(rho_n[0, 0, 3:-3, 3:-3] + rho_n[0, 1, 3:-3, 3:-3]).reshape(256, 256),
                np.linspace(-0.01, 0.01, 100)
               )

    # pl.contourf(np.array(nls.q1_center[:, :, 3:-3, 3:-3]).reshape(256, 256), 
    #             np.array(nls.q2_center[:, :, 3:-3, 3:-3]).reshape(256, 256), 
    #             np.array(rho_n[0, 1, 3:-3, 3:-3]).reshape(256, 256), 100,
    #             alpha = 0.5
    #            )

    pl.colorbar()
    pl.gca().set_aspect('equal')
    pl.title('Time = %.2f'%(t0-dt))
    pl.savefig('images/%04d'%time_index + '.png')
    pl.clf()

    rho_n       = af.sum(rho_n, 1)

    nls.strang_timestep(dt)

    rho_n_plus_one       = -1 * nls.compute_moments('density')
    rho_n_plus_one[0, 1] = -1 * rho_n_plus_one[0, 1]
    rho_n_plus_one       = af.sum(rho_n_plus_one, 1)

    divE = nls.fields_solver.compute_divE()

    divE_minus_rho = np.array(af.abs((divE - rho_n_plus_one)[:, :, 3:-3, 3:-3])).reshape(256, 256)

    drho_dt = (rho_n_plus_one - rho_n) / dt

    J1 = nls.fields_solver.J1
    J2 = nls.fields_solver.J2

    J1_plus_q1 = af.shift(nls.fields_solver.J1, 0, 0, -1)
    J2_plus_q2 = af.shift(nls.fields_solver.J2, 0, 0, 0, -1)

    divJ = (J1_plus_q1 - J1) / nls.dq1 + (J2_plus_q2 - J2) / nls.dq2

    continuity[time_index + 1] = af.mean(af.abs(drho_dt + divJ))
    gauss_law[time_index + 1]  = np.mean(divE_minus_rho)
    
    print(continuity[time_index + 1])
    print(gauss_law[time_index + 1])
    print()

    pl.contourf(divE_minus_rho, 100)
    pl.colorbar()
    pl.savefig('images/%04d'%time_index + '.png')
    pl.clf()

# pl.semilogy(time_array, continuity)
# pl.ylabel(r'$|\frac{d \rho}{d t} + \nabla \cdot \vec{J}|$')
# pl.xlabel('Time')
# pl.savefig('continuity.png')
# pl.clf()

# pl.semilogy(time_array, gauss_law)
# pl.ylabel(r'$|\nabla \cdot \vec{E} - \rho|$')
# pl.xlabel('Time')
# pl.savefig('gauss_law.png')
# pl.clf()
