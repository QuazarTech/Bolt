#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
In this test we check that the 2D Poisson solver
works as intended. For this purpose, we assign
a density distribution for which the analytical
solution for electrostatic fields may be computed.
This solution is then checked against the solution
given by the FFT solver
"""

import numpy as np
import arrayfire as af
from petsc4py import PETSc

from bolt.lib.nonlinear.fields.electrostatic.fft import fft_poisson
from bolt.lib.utils.calculate_q import calculate_q_center

class test(object):
    def __init__(self):

        self.q1_start = 0
        self.q2_start = 0

        self.q1_end = 1
        self.q2_end = 1

        self.N_q1 = 1024
        self.N_q2 = 1024

        self.dq1 = (self.q1_end - self.q1_start) / self.N_q1
        self.dq2 = (self.q2_end - self.q2_start) / self.N_q2

        self.N_ghost = self.N_g = np.random.randint(3, 5)

        self.q1_center, self.q2_center = \
            calculate_q_center(self.q1_start, self.q2_start,
                               self.N_q1, self.N_q2, self.N_ghost,
                               self.dq1, self.dq2
                              )

        self.params = type('obj', (object, ), {'eps':1})

        self.cell_centered_EM_fields = af.constant(0, 6, 1, 
                                                   self.q1_center.shape[2],
                                                   self.q1_center.shape[3],
                                                   dtype=af.Dtype.f64
                                                  )

        self._comm   = PETSc.COMM_WORLD.tompi4py()

        self.performance_test_flag = False

def test_fft_poisson():
    
    obj = test()
    fft_poisson(obj, -af.sin(2 * np.pi * obj.q1_center + 4 * np.pi * obj.q2_center))

    E1_expected = (0.1 / np.pi) * af.cos(  2 * np.pi * obj.q1_center
                                         + 4 * np.pi * obj.q2_center
                                        )

    E2_expected = (0.2 / np.pi) * af.cos(  2 * np.pi * obj.q1_center
                                         + 4 * np.pi * obj.q2_center
                                        )

    N_g = obj.N_ghost

    error_E1 = af.mean(af.abs(  obj.cell_centered_EM_fields[0, 0, N_g:-N_g, N_g:-N_g] 
                              - E1_expected[0, 0, N_g:-N_g, N_g:-N_g]
                             )
                      )

    error_E2 = af.mean(af.abs(  obj.cell_centered_EM_fields[1, 0, N_g:-N_g, N_g:-N_g] 
                              - E2_expected[0, 0, N_g:-N_g, N_g:-N_g]
                             )
                      )

    assert (error_E1 < 1e-14 and error_E2 < 1e-14)
