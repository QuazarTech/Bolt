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

omega = np.sqrt(params.temperature_background * params.gamma) * params.k_q1 * 1j

# Defining the functions for the analytical solution:
def n_ana(q1, t):
    
    n_b = params.density_background

    pert_real_n = 1
    pert_imag_n = 0
    pert_n      = pert_real_n + 1j * pert_imag_n

    n_ana       = n_b + params.amplitude * pert_n * \
                        np.exp(  1j * params.k_q1 * q1 
                               + omega * t
                              ).real

    return(n_ana)

def v1_ana(q1, t):
    
    v1_b = params.v1_bulk_background
    n_b  = params.density_background

    pert_real_v1 = -np.sqrt(params.gamma * T_b) / n_b
    pert_imag_v1 = 0
    pert_v1      = pert_real_v1 + 1j * pert_imag_v1
                   
    v1_ana = v1_b + params.amplitude * pert_v1 * \
                    np.exp(  1j * params.k_q1 * q1 
                           + params.omega * t
                          ).real
    return(v1_ana)

def T_ana(q1, t):
    
    T_b = params.temperature_background
    n_b = params.density_background

    pert_real_T = T_b * (params.gamma - 1) / n_b
    pert_imag_T = 0
    pert_T      = pert_real_T + 1j * pert_imag_T

    T_ana = T_b + params.amplitude * pert_T * \
                  np.exp(  1j * params.k_q1 * q1 
                         + params.omega * t
                        ).real

    return(T_ana)

N        = 2**np.arange(5, 10)
error_n  = np.zeros(N.size)
error_v1 = np.zeros(N.size)
error_T  = np.zeros(N.size)

for i in range(N.size):
    domain.N_q1 = int(N[i])
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

    time_array = np.arange(0, params.t_final + dt, dt)

    for time_index, t0 in enumerate(time_array[1:]):
        print('Computing For Time =', t0)
        nls.strang_timestep(dt)

    # Performing f = f0 at final time:
    self.f = nls._source(self.f, self.time_elapsed,
                         self.q1_center, self.q2_center,
                         self.p1_center, self.p2_center, self.p3_center, 
                         self.compute_moments, 
                         self.physical_system.params, 
                         True
                        )

    n_nls  = nls.compute_moments('density')
    v1_nls = nls.compute_moments('mom_v1_bulk') / n_nls
    v2_nls = nls.compute_moments('mom_v2_bulk') / n_nls
    v3_nls = nls.compute_moments('mom_v3_bulk') / n_nls
    T_nls  = (1 / params.p_dim) * (  2 * nls.compute_moments('energy') 
                                   - n_nls * v1_nls**2
                                   - n_nls * v2_nls**2
                                   - n_nls * v3_nls**2
                                  ) / n_nls

    n_analytic  = n_ana(np.array(nls.q1_center), t0)
    v1_analytic = v1_ana(np.array(nls.q1_center), t0)
    T_analytic  = T_ana(np.array(nls.q1_center), t0)

    error_n[i] = np.mean(abs(  np.array(n_nls)[:, :, N_g_q:-N_g_q] 
                             - n_analytic[:, :, N_g_q:-N_g_q]
                            )
                        )

    error_v1[i] = np.mean(abs(  np.array(v1_nls)[:, :, N_g_q:-N_g_q] 
                              - v1_analytic[:, :, N_g_q:-N_g_q]
                             )
                         )

    error_T[i] = np.mean(abs(  np.array(T_nls)[:, :, N_g_q:-N_g_q] 
                             - T_analytic[:, :, N_g_q:-N_g_q]
                            )
                        )

print(error_n)
print(error_v1)
print(error_T)

print(np.polyfit(np.log10(N), np.log10(error_n), 1))
print(np.polyfit(np.log10(N), np.log10(error_v1), 1))
print(np.polyfit(np.log10(N), np.log10(error_T), 1))

pl.loglog(N, error_n, '-o', label = 'Density')
pl.loglog(N, error_v1, '-o', label = 'Velocity')
pl.loglog(N, error_T, '-o', label = 'Temperature')
pl.loglog(N, 1/N**2, '--', color = 'black', label = r'$O(N^{-2})$')
pl.xlabel(r'$N$')
pl.ylabel('Error')
pl.legend()
pl.savefig('convergenceplot.png')
