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

from lib.nonlinear_solver.EM_fields_solver.electrostatic import fft_poisson
from lib.nonlinear_solver.communicate import communicate_fields


class test(object):
    def __init__(self):

        self.physical_system = type('obj', (object, ),
                            {'params': type('obj', (object, ),
                                {'charge_electron': -1})
                             })

        self.q1_start = 0
        self.q2_start = 0

        self.q1_end = 1
        self.q2_end = 1

        self.N_q1 = np.random.randint(24, 48)
        self.N_q2 = np.random.randint(24, 48)

        self.dq1 = (self.q1_end - self.q1_start) / self.N_q1
        self.dq2 = (self.q2_end - self.q2_start) / self.N_q2

        self.N_ghost = np.random.randint(3, 5)

        self.q1 = self.q1_start + \
                  (0.5 + np.arange(-self.N_ghost,
                                    self.N_q1 + self.N_ghost)) * self.dq1
        self.q2 = self.q2_start + \
                  (0.5 + np.arange(-self.N_ghost,
                                    self.N_q2 + self.N_ghost)) * self.dq2

        self.q2, self.q1 = np.meshgrid(self.q2, self.q1)
        self.q2, self.q1 = af.to_array(self.q2), af.to_array(self.q1)

        # Assigning initial values to zero:
        self.E1 = af.constant(0, self.q1.shape[0], self.q1.shape[1],
                              dtype=af.Dtype.f64)
        self.E2 = self.E1.copy()
        self.E3 = self.E1.copy()

        self.B1 = self.E1.copy()
        self.B2 = self.E1.copy()
        self.B3 = self.E1.copy()

        self._comm = PETSc.COMM_WORLD.tompi4py()

        self._da_fields = PETSc.DMDA().create([self.N_q1, self.N_q2],
                                              dof=6,
                                              stencil_width=self.N_ghost,
                                              boundary_type=('periodic',
                                                             'periodic'),
                                              stencil_type=1, )

        self._glob_fields = self._da_fields.createGlobalVec()
        self._local_fields = self._da_fields.createLocalVec()

    def compute_moments(self, *args):
        return (af.sin(2 * np.pi * self.q1 + 4 * np.pi * self.q2))

    _communicate_fields = communicate_fields


def test_fft_poisson():
    obj = test()
    fft_poisson(obj)

    E1_expected = (0.1 / np.pi) * af.cos(2 * np.pi * obj.q1 +
                                         4 * np.pi * obj.q2)

    E2_expected = (0.2 / np.pi) * af.cos(2 * np.pi * obj.q1 +
                                         4 * np.pi * obj.q2)

    error_E1 = af.sum(af.abs(obj.E1 - E1_expected)) / (obj.E1.elements())
    error_E2 = af.sum(af.abs(obj.E2 - E2_expected)) / (obj.E2.elements())

    assert (error_E1 < 1e-14 and error_E2 < 1e-14)
