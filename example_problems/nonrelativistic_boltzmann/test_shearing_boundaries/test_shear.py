import arrayfire as af
import numpy as np
import pylab as pl
pl.style.use('prettyplot')

from bolt.lib.physical_system import physical_system
from bolt.lib.nonlinear_solver.nonlinear_solver import nonlinear_solver

import domain
import params

import boundary_conditions_x
import initialize_x

import boundary_conditions_y
import initialize_y

import bolt.src.nonrelativistic_boltzmann.advection_terms as advection_terms
import bolt.src.nonrelativistic_boltzmann.collision_operator as collision_operator
import bolt.src.nonrelativistic_boltzmann.moment_defs as moment_defs

def test_shear_x():

    t     = np.random.rand(1)[0]
    N     = 2**np.arange(5, 10)
    error = np.zeros(N.size)

    for i in range(N.size):
        domain.N_q1 = int(N[i])
        domain.N_q2 = int(N[i])

        # Defining the physical system to be solved:
        system = physical_system(domain,
                                 boundary_conditions_x,
                                 params,
                                 initialize_x,
                                 advection_terms,
                                 collision_operator.BGK,
                                 moment_defs
                                )

        L_q1 = domain.q1_end - domain.q1_start
        L_q2 = domain.q2_end - domain.q2_start

        nls = nonlinear_solver(system)
        N_g = nls.N_ghost_q

        # For left:
        f_reference_left = af.broadcast(initialize_x.initialize_f, 
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
        f_reference_right = af.broadcast(initialize_x.initialize_f, 
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

        error[i] =   af.mean(af.abs(nls.f[:, :N_g,  N_g:-N_g] - f_reference_left)) \
                   + af.mean(af.abs(nls.f[:, -N_g:, N_g:-N_g] - f_reference_right))
    
    pl.loglog(N, error, '-o', label = 'Numerical')
    pl.loglog(N, error[0]*32**3/N**3, '--', color = 'black', label = r'$O(N^{-3})$')
    pl.xlabel(r'$N$')
    pl.ylabel('Error')
    pl.legend()
    pl.savefig('plot1.png')

    poly = np.polyfit(np.log10(N), np.log10(error), 1) 
    assert(abs(poly[0]+3)<0.3)

def test_shear_y():
    
    t     = np.random.rand(1)[0]
    N     = 2**np.arange(5, 10)
    error = np.zeros(N.size)

    for i in range(N.size):
        domain.N_q1 = int(N[i])
        domain.N_q2 = int(N[i])

        # Defining the physical system to be solved:
        system = physical_system(domain,
                                 boundary_conditions_y,
                                 params,
                                 initialize_y,
                                 advection_terms,
                                 collision_operator.BGK,
                                 moment_defs
                                )

        L_q1 = domain.q1_end - domain.q1_start
        L_q2 = domain.q2_end - domain.q2_start

        nls = nonlinear_solver(system)
        N_g = nls.N_ghost_q

        # For left:
        f_reference_bot = af.broadcast(initialize_y.initialize_f, 
                                       af.select(nls.q1_center - t<domain.q1_start,   # Periodic domain
                                                 nls.q1_center - t + L_q1,
                                                 nls.q1_center - t
                                                ),
                                       nls.q2_center, 
                                       nls.p1_center, 
                                       nls.p2_center,
                                       nls.p3_center, 
                                       params
                                      )[:, N_g:-N_g, -2 * N_g:-N_g]

        # For right:
        f_reference_top = af.broadcast(initialize_y.initialize_f, 
                                       af.select(nls.q1_center + t>domain.q1_end,
                                                 nls.q1_center + t - L_q1,
                                                 nls.q1_center + t
                                                ),
                                       nls.q2_center, 
                                       nls.p1_center, 
                                       nls.p2_center,
                                       nls.p3_center, 
                                       params
                                      )[:, N_g:-N_g, N_g:2 * N_g]

        nls.time_elapsed = t
        nls._communicate_f()
        nls._apply_bcs_f()

        error[i] =   af.mean(af.abs(nls.f[:, N_g:-N_g, :N_g] - f_reference_bot)) \
                   + af.mean(af.abs(nls.f[:, N_g:-N_g, -N_g:] - f_reference_top))
    
    pl.loglog(N, error, '-o', label = 'Numerical')
    pl.loglog(N, error[0]*32**3/N**3, '--', color = 'black', label = r'$O(N^{-3})$')
    pl.xlabel(r'$N$')
    pl.ylabel('Error')
    pl.legend()
    pl.savefig('plot2.png')

    poly = np.polyfit(np.log10(N), np.log10(error), 1) 
    assert(abs(poly[0]+3)<0.3)
