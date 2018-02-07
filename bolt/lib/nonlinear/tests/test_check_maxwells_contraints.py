#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This checks that the explicit time-stepping of the
FDTD algorithm works as intended. Since Maxwell's
equation have wave like solutions, in this test we evolve
the initial state for a single timeperiod and compare the
final solution state with the initial state.

We check the fall off in error with the increase in resolution
(convergence rate) to validate the explicit FDTD algorithm.
"""

import numpy as np
import arrayfire as af
from petsc4py import PETSc

from bolt.lib.nonlinear.fields.fields import fields_solver
from bolt.lib.physical_system import physical_system
from bolt.lib.utils.calculate_q import calculate_q_center

from input_files import domain
from input_files import params_check_maxwells_contraints
from input_files import initialize_check_maxwells_contraints
from input_files import boundary_conditions

import bolt.src.nonrelativistic_boltzmann.advection_terms as advection_terms
import bolt.src.nonrelativistic_boltzmann.collision_operator as collision_operator
import bolt.src.nonrelativistic_boltzmann.moments as moments

class test(object):

    def __init__(self, N):

        domain.N_q1 = int(N)
        domain.N_q2 = int(N)

        system = physical_system(domain,
                                 boundary_conditions,
                                 params_check_maxwells_contraints,
                                 initialize_check_maxwells_contraints,
                                 advection_terms,
                                 collision_operator.BGK,
                                 moments
                                )

        self.fields_solver = fields_solver(system, 
                                           None, 
                                           False
                                          )

        return

def test_check_maxwells_constraints():

    params = params_check_maxwells_contraints
    system = physical_system(domain,
                             boundary_conditions,
                             params,
                             initialize_check_maxwells_contraints,
                             advection_terms,
                             collision_operator.BGK,
                             moments
                            )

    dq1 = (domain.q1_end - domain.q1_start) / domain.N_q1
    dq2 = (domain.q2_end - domain.q2_start) / domain.N_q2
    
    q1, q2 = calculate_q_center(domain.q1_start, domain.q2_start,
                                domain.N_q1, domain.N_q2, domain.N_ghost,
                                dq1, dq2
                               )
    
    rho = (  params.pert_real * af.cos(  params.k_q1 * q1 
                                       + params.k_q2 * q2
                                      )
           - params.pert_imag * af.sin(  params.k_q1 * q1 
                                       + params.k_q2 * q2
                                      )
          )

    obj = fields_solver(system, 
                        rho, 
                        False
                       )

    # Checking for ∇.E = rho / epsilon
    rho_left_bot = 0.25 * (  rho 
                           + af.shift(rho, 0, 0, 0, 1)
                           + af.shift(rho, 0, 0, 1, 0)
                           + af.shift(rho, 0, 0, 1, 1)
                          ) 

    N_g = obj.N_g
    assert(af.mean(af.abs(obj.compute_divB()[:, :, N_g:-N_g, N_g:-N_g]))<1e-14)

    divE  = obj.compute_divE()
    rho_b = af.mean(rho_left_bot) # background

    assert(af.mean(af.abs(divE - rho_left_bot + rho_b)[:, :, N_g:-N_g, N_g:-N_g])<1e-6)
