import arrayfire as af
import numpy as np
import pylab as pl
pl.style.use('prettyplot')

from bolt.lib.physical_system import physical_system
from bolt.lib.nonlinear_solver.nonlinear_solver import nonlinear_solver

import domain
import boundary_conditions
import params
import initialize

import bolt.src.nonrelativistic_boltzmann.advection_terms as advection_terms
import bolt.src.nonrelativistic_boltzmann.collision_operator as collision_operator
import bolt.src.nonrelativistic_boltzmann.moment_defs as moment_defs

t     = 0.05 #np.random.rand(1)[0]
N     = 2**np.arange(5, 10)
error = np.zeros(N.size)

for i in range(N.size):
    domain.N_q1 = int(N[i])
    domain.N_q2 = int(N[i])

    # Defining the physical system to be solved:
    system = physical_system(domain,
                             boundary_conditions,
                             params,
                             initialize,
                             advection_terms,
                             collision_operator.BGK,
                             moment_defs
                            )

    L_q1 = domain.q1_end - domain.q1_start
    L_q2 = domain.q2_end - domain.q2_start

    nls = nonlinear_solver(system)
    N_g = nls.N_ghost_q

    # For left:
    f_reference_left = af.broadcast(initialize.initialize_f, 
                                    nls.q1_center, 
                                    af.select(nls.q2_center - t<domain.q2_start,   # Periodic domain
                                              nls.q2_center - t + L_q2,
                                              nls.q2_center - t
                                             ),
                                    nls.p1_center, 
                                    nls.p2_center,
                                    nls.p3_center, 
                                    params
                                   )[:, -2 * N_g:-N_g, N_g:-N_g]

    # For right:
    f_reference_right = af.broadcast(initialize.initialize_f, 
                                     nls.q1_center, 
                                     af.select(nls.q2_center + t>domain.q2_end,
                                               nls.q2_center + t - L_q2,
                                               nls.q2_center + t
                                              ),
                                     nls.p1_center, 
                                     nls.p2_center,
                                     nls.p3_center, 
                                     params
                                    )[:, N_g:2 * N_g, N_g:-N_g]

    nls.time_elapsed = t
    nls._communicate_f()
    nls._apply_bcs_f()

    error[i] = af.mean(af.abs(nls.f[:, :N_g,  N_g:-N_g] - f_reference_left))
    
    # pl.contourf(np.array(af.reorder(nls.f[:, :N_g,  N_g:-N_g], 1, 2, 0)), 100)
    # pl.colorbar()
    # pl.show()

    # pl.contourf(np.array(af.reorder(f_reference_left, 1, 2, 0)), 100)
    # pl.colorbar()
    # pl.show()

    # pl.contourf(np.array(af.reorder(af.abs(f_reference_left-nls.f[:, :N_g,  N_g:-N_g]), 1, 2, 0)), 100)
    # pl.colorbar()
    # pl.show()

    # print(af.mean(af.abs(nls.f[:, -N_g:, N_g:-N_g] - f_reference_right)))

print(error)
pl.loglog(N, error, '-o', label = 'Numerical')
pl.loglog(N, error[0]*32/N, '--', color = 'black', label = r'$O(N^{-1})$')
pl.xlabel(r'$N$')
pl.ylabel('Error')
pl.legend()

pl.savefig('plot.png')
print(np.polyfit(np.log10(N), np.log10(error), 1))
