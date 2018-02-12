import arrayfire as af
import numpy as np
import h5py
import pylab as pl

from bolt.lib.physical_system import physical_system
from bolt.lib.nonlinear_solver.nonlinear_solver import nonlinear_solver

import domain
import boundary_conditions
import params
import initialize

import bolt.src.nonrelativistic_boltzmann.advection_terms as advection_terms
import bolt.src.nonrelativistic_boltzmann.collision_operator as collision_operator
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

@af.broadcast
def addition(a, b):
    return(a+b)

N = 2**np.arange(5, 10)

def check_error(params):
    error = np.zeros(N.size)

    for i in range(N.size):
        domain.N_p1 = int(N[i])
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
        N_g = nls.N_ghost_q

        # Time parameters:
        dt      = 0.01 * 32/nls.N_p1
        t_final = 0.2

        time_array  = np.arange(dt, t_final + dt, dt)

        # Since only the p = 1 mode is excited:

        E1 = nls.cell_centered_EM_fields_at_n[0]
        E2 = nls.cell_centered_EM_fields_at_n[1]
        E3 = nls.cell_centered_EM_fields_at_n[2]

        B1 = nls.cell_centered_EM_fields_at_n[3]
        B2 = nls.cell_centered_EM_fields_at_n[4]
        B3 = nls.cell_centered_EM_fields_at_n[5]

        (A_p1, A_p2, A_p3) = af.broadcast(nls._A_p, nls.q1_center, nls.q2_center,
                                          nls.p1_center, nls.p2_center, nls.p3_center,
                                          E1, E2, E3, B1, B2, B3,
                                          nls.physical_system.params
                                         )

        f_analytic = af.broadcast(initialize.initialize_f, nls.q1_center, nls.q2_center,
                             addition(nls.p1_center, - A_p1 * t_final), 
                             addition(nls.p2_center, - A_p2 * t_final),
                             nls.p3_center, nls.physical_system.params
                            )

        for time_index, t0 in enumerate(time_array):
            nls.strang_timestep(dt)

        error[i] = af.mean(af.abs(  nls.f
                                  - f_analytic
                                 )
                          )

    return(error)

params.solver_method_in_p         = 'FVM'
params.riemann_solver_in_p        = 'upwind-flux'
params.reconstruction_method_in_p = 'weno5'

weno5_err = check_error(params)
weno5_con = np.polyfit(np.log10(N), np.log10(weno5_err), 1)

params.reconstruction_method_in_p = 'ppm'

ppm_err = check_error(params)
ppm_con = np.polyfit(np.log10(N), np.log10(ppm_err), 1)

params.reconstruction_method_in_p = 'minmod'

minmod_err = check_error(params)
minmod_con = np.polyfit(np.log10(N), np.log10(minmod_err), 1)

params.reconstruction_method_in_p = 'piecewise-constant'

pc_err = check_error(params)
pc_con = np.polyfit(np.log10(N), np.log10(pc_err), 1)

print(weno5_err)
print(ppm_err)
print(minmod_err)
print(pc_err)

print('Convergence with WENO5 reconstruction:', weno5_con[0])
print('Convergence with PPM reconstruction:', ppm_con[0])
print('Convergence with minmod reconstruction:', minmod_con[0])
print('Convergence with piecewise-constant reconstruction:', pc_con[0])

pl.loglog(N, weno5_err, '-o',label = 'WENO5')
pl.loglog(N, ppm_err, '-o', label = 'PPM')
pl.loglog(N, minmod_err, '-o', label = 'minmod')
pl.loglog(N, pc_err, '-o', label = 'Piecewise-Constant')
pl.loglog(N, 1e-3/N, '--', color = 'black', label = r'$O(N^{-1})$')
pl.loglog(N, 1e-2/N**2, '-.', color = 'black', label = r'$O(N^{-2})$')
pl.xlabel(r'$N$')
pl.ylabel('Error')
pl.legend()
pl.title('With Upwind-Flux Riemann Solver')
pl.savefig('convergenceplot.png')
