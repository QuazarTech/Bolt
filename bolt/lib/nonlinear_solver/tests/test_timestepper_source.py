#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This test ensures that the implementation of time-steppers is accurate
For this, we consider the test problem df/dt = f
We integrate till t = 1 and compare the results with the expected
analytic solution f = e^t
"""

import numpy as np
import arrayfire as af

from bolt.lib.nonlinear_solver.timestepper_source \
    import RK2_step, RK4_step, RK6_step


class test(object):
    def __init__(self):
        self.f = af.to_array(np.array([1.0]))
        self.performance_test_flag = False
        self.testing_source_flag   = True

    def _source(self, f):
        return (f)


# This test ensures that the RK2 implementation is 2nd order in time
def test_RK2():
    number_of_time_step = 10**np.arange(5)
    time_step_sizes = 1 / number_of_time_step
    error = np.zeros(time_step_sizes.size)

    for i in range(time_step_sizes.size):
        test_obj = test()
        for j in range(number_of_time_step[i]):
            RK2_step(test_obj, time_step_sizes[i])
        error[i] = abs(af.sum(test_obj.f) - np.exp(1))

    poly = np.polyfit(np.log10(number_of_time_step), np.log10(error), 1)
    assert (abs(poly[0] + 2) < 0.2)


# This test ensures that the RK4 implementation is 4th order in time
def test_RK4():
    number_of_time_step = 10**np.arange(4)
    time_step_sizes = 1 / number_of_time_step
    error = np.zeros(time_step_sizes.size)

    for i in range(time_step_sizes.size):
        test_obj = test()
        for j in range(number_of_time_step[i]):
            RK4_step(test_obj, time_step_sizes[i])
        error[i] = abs(af.sum(test_obj.f) - np.exp(1))

    poly = np.polyfit(np.log10(number_of_time_step), np.log10(error), 1)
    assert (abs(poly[0] + 4) < 0.2)


# This test ensures that the RK6 implementation is 5th order in time
def test_RK6():
    number_of_time_step = 10**np.arange(3)
    time_step_sizes = 1 / number_of_time_step
    error = np.zeros(time_step_sizes.size)

    for i in range(time_step_sizes.size):
        test_obj = test()
        for j in range(number_of_time_step[i]):
            RK6_step(test_obj, time_step_sizes[i])
        error[i] = abs(af.sum(test_obj.f) - np.exp(1))

    poly = np.polyfit(np.log10(number_of_time_step), np.log10(error), 1)
    assert (abs(poly[0] + 5) < 0.2)
