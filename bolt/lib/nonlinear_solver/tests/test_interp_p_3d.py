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

        p1_center = \
            self.p1_start + (0.5 + np.arange(0, self.N_p1, 1)) * self.dp1
        p2_center = \
            self.p2_start + (0.5 + np.arange(0, self.N_p2, 1)) * self.dp2
        p3_center = \
            self.p3_start + (0.5 + np.arange(0, self.N_p3, 1)) * self.dp3

        p2_center, p1_center, p3_center = np.meshgrid(p2_center,
                                                      p1_center,
                                                      p3_center
                                                     )

        p1_center = af.flat(af.to_array(p1_center))
        p2_center = af.flat(af.to_array(p2_center))
        p3_center = af.flat(af.to_array(p3_center))

        self.p1 = af.reorder(p1_center, 2, 3, 0, 1)
        self.p2 = af.reorder(p2_center, 2, 3, 0, 1)
        self.p3 = af.reorder(p3_center, 2, 3, 0, 1)

        self.q1_center = self.q2_center = np.random.rand(1)

        # Creating Dummy Values:
        self.E1 = self.q1_center
        self.E2 = self.q1_center
        self.E3 = self.q1_center

        self.B1_n = self.q1_center
        self.B2_n = self.q1_center
        self.B3_n = self.q1_center

        self.f = af.exp(-self.p1**2 - 2*self.p2**2 - 3*self.p3**2)

        self.physical_system = type('obj', (object, ),
                                    {'params': 'placeHolder'}
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
    N     = 2**np.arange(5, 9)#np.array([16, 24, 32, 48, 64, 80, 96, 112, 128])
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
