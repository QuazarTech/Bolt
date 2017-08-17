#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
In this test, we ensure that the value returned by the function
calculate_q_center() in nonlinear_solver is consistent with our expected
results. Although both the values checked against and the generated values
are essentially the same formulation, the failure of this test may indicate
any accidental changes that may have been introduced.
"""

import numpy as np
import arrayfire as af
from petsc4py import PETSc

from bolt.lib.nonlinear_solver.nonlinear_solver import nonlinear_solver

calculate_q_center = nonlinear_solver._calculate_q_center


class test(object):
    def __init__(self):
        self.q1_start = np.random.randint(0, 5)
        self.q2_start = np.random.randint(0, 5)

        self.q1_end = np.random.randint(5, 10)
        self.q2_end = np.random.randint(5, 10)

        self.N_q1 = np.random.randint(16, 32)
        self.N_q2 = np.random.randint(16, 32)

        self.dq1 = (self.q1_end - self.q1_start) / self.N_q1
        self.dq2 = (self.q2_end - self.q2_start) / self.N_q2

        self.N_ghost = np.random.randint(1, 5)

        self._da = PETSc.DMDA().create([self.N_q1, self.N_q2],
                                        dof = 1,
                                        stencil_width=self.N_ghost)


def test_calculate_q():
    obj = test()
    q1, q2 = calculate_q_center(obj)

    q1_expected = obj.q1_start + \
        (0.5 + np.arange(-obj.N_ghost, obj.N_q1 + obj.N_ghost)) * obj.dq1
    q2_expected = obj.q2_start + \
        (0.5 + np.arange(-obj.N_ghost, obj.N_q2 + obj.N_ghost)) * obj.dq2

    q1_expected = af.tile(af.to_array(q1_expected), 1,
                          obj.N_q2 + 2 * obj.N_ghost,
                          )

    q2_expected = af.tile(af.reorder(af.to_array(q2_expected)),
                          obj.N_q1 + 2 * obj.N_ghost, 1,
                          )

    assert (af.sum(af.abs(q1_expected - q1)) +
            af.sum(af.abs(q2_expected - q2)) == 0)
