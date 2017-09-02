#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The code needs to handle uptil 5D phase space. However, ArrayFire
sets a hard limit on the dimensionality of the array to be 4.

To surpass this, we internally use 2 different forms in which the array
can be stored:
- qExpanded - shape:(Nq1, Nq2, Np1 * Np2 * Np3)
- pExpanded - shape:(Nq1 * Nq2, Np1, Np2, Np3)

This file contains the test functions that are used to convert the arrays
from one from to another and viceversa
"""


import numpy as np
import arrayfire as af
from petsc4py import PETSc

from bolt.lib.nonlinear_solver.nonlinear_solver import nonlinear_solver

convert_to_pExpanded = nonlinear_solver._convert_to_pExpanded
convert_to_qExpanded = nonlinear_solver._convert_to_qExpanded


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

        self._da_f = PETSc.DMDA().create([self.N_q1, self.N_q2],
                                         dof=(self.N_p1 * self.N_p2 * self.N_p3),
                                         stencil_width=self.N_ghost, )


def test_convert_to_pExpanded():
    obj = test()
    test_array = af.randu(obj.N_q1 + 2 * obj.N_ghost,
                          obj.N_q2 + 2 * obj.N_ghost,
                          obj.N_p1 * obj.N_p2 * obj.N_p3)

    modified = convert_to_pExpanded(obj, test_array)

    expected = af.moddims(test_array,
                          (obj.N_q1 + 2 * obj.N_ghost) *
                          (obj.N_q2 + 2 * obj.N_ghost),
                          obj.N_p1, obj.N_p2, obj.N_p3)

    assert (af.sum(modified - expected) == 0)


def test_convert_to_qExpanded():
    obj = test()
    test_array = af.randu((obj.N_q1 + 2 * obj.N_ghost) *
                          (obj.N_q2 + 2 * obj.N_ghost),
                          obj.N_p1, obj.N_p2, obj.N_p3)

    modified = convert_to_qExpanded(obj, test_array)

    expected = af.moddims(test_array, (obj.N_q1 + 2 * obj.N_ghost),
                          (obj.N_q2 + 2 * obj.N_ghost),
                          obj.N_p1 * obj.N_p2 * obj.N_p3)

    assert (af.sum(modified - expected) == 0)
