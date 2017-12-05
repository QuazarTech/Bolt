#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This test ensures that the convergence for the interpolation routine
falls off with N^{-2}, where N is the number of divisions chosen.
"""

import arrayfire as af
import numpy as np
from petsc4py import PETSc

from bolt.lib.nonlinear_solver.nonlinear_solver import nonlinear_solver
from bolt.lib.nonlinear_solver.interpolation_routines import f_interp_p_3d
from bolt.lib.nonlinear_solver.nonlinear_solver import nonlinear_solver

calculate_q_center = nonlinear_solver._calculate_q_center
calculate_p_center = nonlinear_solver._calculate_p_center

convert_p_imported = nonlinear_solver._convert_to_p_expanded
convert_q_imported = nonlinear_solver._convert_to_q_expanded

class test(object):
    def __init__(self, N):
        self.N_p1 = N
        self.N_p2 = N
        self.N_p3 = N

        self.N_q1    = 1
        self.N_q2    = 1
        self.N_ghost = 0

        self.dp1 = 20 / self.N_p1
        self.dp2 = 20 / self.N_p2
        self.dp3 = 20 / self.N_p3

        self.p1_start = self.p2_start = self.p3_start = -10

        self.p1, self.p2, self.p3 = calculate_p_center(self)

        self.q1_center = self.q2_center = np.random.rand(1)

        # Creating Dummy Values:
        self.cell_centered_EM_fields  = np.random.rand(6)
        self.B_fields_at_nth_timestep = np.random.rand(3)
        self.f = af.exp(-self.p1**2 - 2*self.p2**2 - 3*self.p3**2)

        self.physical_system = type('obj', (object, ),
                                    {'params': type('obj', (object, ),{'p_dim':3})}
                                   )

        self._da_f = PETSc.DMDA().create([self.N_q1, self.N_q2],
                                         dof=(  self.N_p1
                                              * self.N_p2
                                              * self.N_p3
                                             )
                                        )

        self.performance_test_flag = False

    def _A_p(self, *args):
        return (1, 1, 1)

    _convert_to_p_expanded = convert_p_imported
    _convert_to_q_expanded = convert_q_imported


def test_f_interp_p_3d():
    N     = 2**np.arange(5, 9)
    error = np.zeros(N.size)

    for i in range(N.size):
        af.device_gc()
        obj = test(int(N[i]))
        
        f_interp_p_3d(obj, 1e-5)
        
        f_analytic = af.exp(-   (obj.p1 - 1e-5)**2 
                            - 2*(obj.p2 - 1e-5)**2 
                            - 3*(obj.p3 - 1e-5)**2
                           )

        error[i] = af.sum(af.abs(obj.f - f_analytic)) / f_analytic.elements()

    poly = np.polyfit(np.log10(N), np.log10(error), 1)
    assert(abs(poly[0] + 2)<0.2)
