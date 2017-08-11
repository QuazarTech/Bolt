#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from lib.linear_solver.timestepper import RK2_step, RK4_step, RK6_step

# This test ensures that the RK2,4,6 implementation is 2nd, 4th, 5h order in time
# For this, we consider the test problem df/dt = f
# We integrate till t = 1 and compare the results with the expected
# analytic solution f = Ae^t


class test(object):
    def __init__(self):
        self.Y = 100

    def _dY_dt(self, Y):
        return(Y)


def test_RK2():
    number_of_time_step = 10**np.arange(1, 4)
    time_step_sizes = 1 / number_of_time_step
    error = np.zeros(time_step_sizes.size)

    for i in range(time_step_sizes.size):
        test_obj = test()
        for j in range(number_of_time_step[i]):
            RK2_step(test_obj, time_step_sizes[i])
        error[i] = abs(test_obj.Y - 100 * np.exp(1))

    poly = np.polyfit(np.log10(number_of_time_step), np.log10(error), 1)
    print(poly)
    assert(abs(poly[0] + 2) < 0.2)


def test_RK4():
    number_of_time_step = 10**np.arange(1, 4)
    time_step_sizes = 1 / number_of_time_step
    error = np.zeros(time_step_sizes.size)

    for i in range(time_step_sizes.size):
        test_obj = test()
        for j in range(number_of_time_step[i]):
            RK4_step(test_obj, time_step_sizes[i])
        error[i] = abs(test_obj.Y - 100 * np.exp(1))

    poly = np.polyfit(np.log10(number_of_time_step), np.log10(error), 1)
    print(poly)
    assert(abs(poly[0] + 4) < 0.2)


def test_RK6():
    number_of_time_step = 10**np.arange(1, 3)
    time_step_sizes = 1 / number_of_time_step
    error = np.zeros(time_step_sizes.size)

    for i in range(time_step_sizes.size):
        test_obj = test()
        for j in range(number_of_time_step[i]):
            RK6_step(test_obj, time_step_sizes[i])
        error[i] = abs(test_obj.Y - 100 * np.exp(1))

    poly = np.polyfit(np.log10(number_of_time_step), np.log10(error), 1)
    print(poly)
    assert(abs(poly[0] + 5) < 0.2)
