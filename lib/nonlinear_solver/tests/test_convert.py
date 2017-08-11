#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import arrayfire as af
from petsc4py import PETSc

from lib.nonlinear_solver.nonlinear_solver import nonlinear_solver

convert = nonlinear_solver._convert


class test(object):
    def __init__(self):
        self.q1_start = np.random.randint(0, 5)
        self.q2_start = np.random.randint(0, 5)

        self.q1_end = np.random.randint(5, 10)
        self.q2_end = np.random.randint(5, 10)

        self.N_q1 = np.random.randint(16, 32)
        self.N_q2 = np.random.randint(16, 32)

        self.N_ghost = np.random.randint(1, 5)

        self.N_p1 = np.random.randint(16, 32)
        self.N_p2 = np.random.randint(16, 32)
        self.N_p3 = np.random.randint(16, 32)

        self._da = PETSc.DMDA().create(
            [self.N_q1, self.N_q2],
            dof=(self.N_p1 * self.N_p2 * self.N_p3),
            stencil_width=self.N_ghost, )


def test_convert_1():
    obj = test()
    test_array = af.randu(obj.N_q1 + 2 * obj.N_ghost,
                          obj.N_q2 + 2 * obj.N_ghost,
                          obj.N_p1 * obj.N_p2 * obj.N_p3)

    modified = convert(obj, test_array)

    expected = af.moddims(test_array, (obj.N_q1 + 2 * obj.N_ghost) *
                          (obj.N_q2 + 2 * obj.N_ghost), obj.N_p1, obj.N_p2,
                          obj.N_p3)

    assert (af.sum(modified - expected) == 0)


def test_convert_2():
    obj = test()
    test_array = af.randu((obj.N_q1 + 2 * obj.N_ghost) * \
                          (obj.N_q2 + 2 * obj.N_ghost), obj.N_p1, obj.N_p2, obj.N_p3)

    modified = convert(obj, test_array)

    expected = af.moddims(test_array, (obj.N_q1 + 2 * obj.N_ghost),
                          (obj.N_q2 + 2 * obj.N_ghost),
                          obj.N_p1 * obj.N_p2 * obj.N_p3)

    assert (af.sum(modified - expected) == 0)
