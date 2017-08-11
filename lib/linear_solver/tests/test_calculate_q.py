#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importing dependencies:
import numpy as np
import arrayfire as af

# Importing solver functions:
from lib.linear_solver.linear_solver import linear_solver as linear_solver

calculate_q_center = linear_solver._calculate_q_center

# In this test, we ensure that the value returned by the function calculate_q_center() in
# linear_solver is consistent with our expected results. Although both the values checked
# against and the generated values are essentially the same formulation, the failure of this
# test may indicate any accidental changes that may have been introduced.


class test():
    def __init__(self):
        self.q1_start = np.random.randint(0, 5)
        self.q2_start = np.random.randint(0, 5)

        self.q1_end = np.random.randint(5, 10)
        self.q2_end = np.random.randint(5, 10)

        self.N_q1 = np.random.randint(32, 128)
        self.N_q2 = np.random.randint(32, 128)

        self.dq1 = (self.q1_end - self.q1_start) / self.N_q1
        self.dq2 = (self.q2_end - self.q2_start) / self.N_q2

        self.N_p1 = np.random.randint(5, 20)
        self.N_p2 = np.random.randint(5, 20)
        self.N_p3 = np.random.randint(5, 20)


def test_calculate_q_center():
    test_obj = test()

    q1, q2 = calculate_q_center(test_obj)

    q1_expected = test_obj.q1_start + \
        (0.5 + np.arange(test_obj.N_q1)) * test_obj.dq1
    q2_expected = test_obj.q2_start + \
        (0.5 + np.arange(test_obj.N_q2)) * test_obj.dq2

    q1_expected = af.tile(
        af.to_array(q1_expected),
        1,
        test_obj.N_q2,
        test_obj.N_p1 *
        test_obj.N_p2 *
        test_obj.N_p3)
    q2_expected = af.tile(
        af.reorder(
            af.to_array(q2_expected)),
        test_obj.N_q1,
        1,
        test_obj.N_p1 *
        test_obj.N_p2 *
        test_obj.N_p3)

    assert(
        af.sum(
            af.abs(
                q1_expected -
                q1)) +
        af.sum(
            af.abs(
                q2_expected -
                q2)) == 0)
