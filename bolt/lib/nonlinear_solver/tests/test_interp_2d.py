#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This test ensures that the convergence for the interpolation routine
falls off with N^{-2}, where N is the number of divisions chosen.
"""

import arrayfire as af
import numpy as np
from petsc4py import PETSc

from bolt.lib.nonlinear_solver.interpolation_routines \
    import f_interp_2d


class test(object):
    def __init__(self, N_q1, N_q2, N_ghost):
        self._da_f = \
            PETSc.DMDA().create([N_q1, N_q2],
                                stencil_width=N_ghost,
                                boundary_type=('periodic', 'periodic')
                               )

        self.N_q1 = N_q1
        self.N_q2 = N_q2

        self.dq1 = 1 / N_q1
        self.dq2 = 1 / N_q2

        self.N_ghost = N_ghost

        self._A_q1 = 1
        self._A_q2 = 1

        self.q1_start = self.q2_start = 0
        
        self.q1_center = \
            af.to_array((-N_ghost + np.arange(N_q1 + 2 * N_ghost) + 0.5) * (1 / N_q1))
        self.q2_center = \
            af.to_array((-N_ghost + np.arange(N_q2 + 2 * N_ghost) + 0.5) * (1 / N_q2))
        
        self.q1_center = af.tile(self.q1_center, 1, N_q2 + 2 * N_ghost)
        self.q2_center = \
            af.tile(af.reorder(self.q2_center), N_q1 + 2 * N_ghost, 1)

        self.f = af.sin(2 * np.pi * self.q1_center +
                        4 * np.pi * self.q2_center
                       )

        self.performance_test_flag = False

def test_f_interp_2d():
    N = 2**np.arange(5, 11)
    error = np.zeros(N.size)

    for i in range(N.size):
        test_obj = test(int(N[i]), int(N[i]), 3)
        f_interp_2d(test_obj, 0.00001)
        f_analytic = af.sin(2 * np.pi * (test_obj.q1_center - 0.00001) +
                            4 * np.pi * (test_obj.q2_center - 0.00001)
                           )
        error[i] = af.sum(af.abs(test_obj.f[3:-3, 3:-3] - f_analytic[3:-3, 3:-3])) \
                   / f_analytic[3:-3, 3:-3].elements()

    poly = np.polyfit(np.log10(N), np.log10(error), 1)
    assert (abs(poly[0] + 2) < 0.2)
