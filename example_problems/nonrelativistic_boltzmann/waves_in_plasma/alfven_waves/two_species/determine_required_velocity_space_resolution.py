import arrayfire as af
import numpy as np
import pylab as pl
import math
from petsc4py import PETSc

from bolt.lib.physical_system import physical_system
from bolt.lib.nonlinear.nonlinear_solver import nonlinear_solver
from bolt.lib.nonlinear.finite_volume.df_dt_fvm import df_dt_fvm

import domain
import boundary_conditions
import initialize
import params

import bolt.src.nonrelativistic_boltzmann.advection_terms as advection_terms
import bolt.src.nonrelativistic_boltzmann.collision_operator as collision_operator
import bolt.src.nonrelativistic_boltzmann.moments as moments

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 16, 10 #10, 14
pl.rcParams['figure.dpi']      = 80
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

# This segment of code used to determine required velocity space resolution:
# Comment this out once determined:
N          = np.array([32, 48, 64, 96, 128])
error_p2_e = np.zeros(N.size)
error_p2_i = np.zeros(N.size)
error_p3_e = np.zeros(N.size)
error_p3_i = np.zeros(N.size)

for i in range(N.size):
    
    domain.N_p2 = int(N[i])
    domain.N_p3 = int(N[i])
    
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

    # Time parameters:
    dt_fvm = params.N_cfl * min(nls.dq1, nls.dq2) \
                          / max(domain.p1_end + domain.p2_end + domain.p3_end) # joining elements of the list

    dt_fdtd = params.N_cfl * min(nls.dq1, nls.dq2) \
                           / params.c # lightspeed

    dt = min(dt_fvm, dt_fdtd)

    v2_bulk_i =   params.amplitude * -4.801714581503802e-15 * af.cos(params.k_q1 * nls.q1_center) \
                - params.amplitude * -0.3692429960259134 * af.sin(params.k_q1 * nls.q1_center)

    v2_bulk_e =   params.amplitude * -4.85722573273506e-15 * af.cos(params.k_q1 * nls.q1_center) \
                - params.amplitude * - 0.333061857862197* af.sin(params.k_q1 * nls.q1_center)

    v3_bulk_i =   params.amplitude * -0.3692429960259359 * af.cos(params.k_q1 * nls.q1_center) \
                - params.amplitude * 1.8041124150158794e-16  * af.sin(params.k_q1 * nls.q1_center)

    v3_bulk_e =   params.amplitude * -0.333061857862222 * af.cos(params.k_q1 * nls.q1_center) \
                - params.amplitude * -3.885780586188048e-16 * af.sin(params.k_q1 * nls.q1_center)

    # Checking that the velocity space is well-resolved:
    nls.dt = dt
    d_flux_p2_dp2 = df_dt_fvm(nls.f, nls, 'd_flux_p2_dp2')
    d_flux_p3_dp3 = df_dt_fvm(nls.f, nls, 'd_flux_p3_dp3')
    
    numerical_first_moment_p2_term  = nls.compute_moments('mom_v2_bulk', f = d_flux_p2_dp2)
    analytic_first_moment_p2_term_e = \
        10 * (nls.fields_solver.yee_grid_EM_fields[1] + v3_bulk_e * nls.fields_solver.yee_grid_EM_fields[3])
    analytic_first_moment_p2_term_i  = \
        -1 * (nls.fields_solver.yee_grid_EM_fields[1] + v3_bulk_i * nls.fields_solver.yee_grid_EM_fields[3])
    
    # pl.plot(np.array(numerical_first_moment_p2_term[:, 0, :, 0]).ravel(), label = 'N = ' + str(N[i]))
    # pl.plot(np.array(analytic_first_moment_p2_term_e[:, 0, :, 0]).ravel(), '--', color = 'black')
    # pl.show()

    # pl.plot(np.array(numerical_first_moment_p2_term[:, 1, :, 0]).ravel(), label = 'N = ' + str(N[i]))
    # pl.plot(np.array(analytic_first_moment_p2_term_i[:, 0, :, 0]).ravel(), '--', color = 'black')
    # pl.show()

    numerical_first_moment_p3_term  = nls.compute_moments('mom_v3_bulk', f = d_flux_p3_dp3)
    analytic_first_moment_p3_term_e = \
        10 * (nls.fields_solver.yee_grid_EM_fields[2] - v2_bulk_e * nls.fields_solver.yee_grid_EM_fields[3])
    analytic_first_moment_p3_term_i  = \
        -1 * (nls.fields_solver.yee_grid_EM_fields[2] - v2_bulk_i * nls.fields_solver.yee_grid_EM_fields[3])

    # pl.plot(np.array(numerical_first_moment_p3_term[:, 0, :, 0]).ravel(), label = 'N = ' + str(N[i]))
    # pl.plot(np.array(analytic_first_moment_p3_term_e[:, 0, :, 0]).ravel(), '--', color = 'black')
    # pl.show()

    # pl.plot(np.array(numerical_first_moment_p3_term[:, 1, :, 0]).ravel(), label = 'N = ' + str(N[i]))
    # pl.plot(np.array(analytic_first_moment_p3_term_i[:, 0, :, 0]).ravel(), '--', color = 'black')
    # pl.show()

    error_p2_e[i] = af.mean(af.abs(numerical_first_moment_p2_term[:, 0] - analytic_first_moment_p2_term_e))
    error_p2_i[i] = af.mean(af.abs(numerical_first_moment_p2_term[:, 1] - analytic_first_moment_p2_term_i))
    error_p3_e[i] = af.mean(af.abs(numerical_first_moment_p3_term[:, 0] - analytic_first_moment_p3_term_e))
    error_p3_i[i] = af.mean(af.abs(numerical_first_moment_p3_term[:, 1] - analytic_first_moment_p3_term_i))

pl.loglog(N, error_p2_e, '-o', label = r'$p_{2e}$')
pl.loglog(N, error_p2_i, '-o', label = r'$p_{2i}$')
pl.loglog(N, error_p3_e, '-o', label = r'$p_{3e}$')
pl.loglog(N, error_p3_i, '-o', label = r'$p_{3i}$')
pl.loglog(N, error_p2_e[0] * 32**2 / N**2, '--', color = 'black', label = r'$\mathcal{O}(N^2)$')
pl.xlabel(r'$N$')
pl.ylabel('Error')
pl.legend()
pl.savefig('convergenceplot.png')
