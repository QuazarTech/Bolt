#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This test ensures that the implementation of splitting schemes
is consistent with the expected results.
"""

import arrayfire as af
import numpy as np

from bolt.lib.nonlinear_solver.temporal_evolution.operator_splitting_methods \
    import strang, lie, swss, jia
from bolt.lib.nonlinear_solver.temporal_evolution.integrators import RK4

class test(object):
    def __init__(self):
        self.f = af.to_array(np.array([0]))

        self.cell_centered_EM_fields = af.to_array(np.array([0]))
        self.yee_grid_EM_fields      = af.to_array(np.array([0]))

        self.performance_test_flag = False

    def df_dt_1(self, f):
        return (f)

    def df_dt_2(self, f):
        return (1)

def op1(self, dt):
    self.f = RK4(self.df_dt_1, self.f, dt)
    return

def op2(self, dt):
    self.f = RK4(self.df_dt_2, self.f, dt)
    return

def test_lie_split_operations():

    number_of_time_step = 10**np.arange(3)
    time_step_sizes     = 1 / number_of_time_step

    error = np.zeros(time_step_sizes.size)

    for i in range(time_step_sizes.size):
        test_obj = test()
        for j in range(number_of_time_step[i]):
            lie(test_obj, op1, op2, time_step_sizes[i])
        error[i] = abs(af.sum(test_obj.f) - np.exp(1) + 1)

    poly = np.polyfit(np.log10(number_of_time_step), np.log10(error), 1)
    assert (abs(poly[0] + 1) < 0.2)

def test_strang_split_operations():

    number_of_time_step = 10**np.arange(3)
    time_step_sizes     = 1 / number_of_time_step
    error = np.zeros(time_step_sizes.size)

    for i in range(time_step_sizes.size):
        test_obj = test()
        for j in range(number_of_time_step[i]):
            strang(test_obj, op1, op2, time_step_sizes[i])
        error[i] = abs(af.sum(test_obj.f) - np.exp(1) + 1)

    poly = np.polyfit(np.log10(number_of_time_step), np.log10(error), 1)
    assert (abs(poly[0] + 2) < 0.2)

def test_swss_split_operations():

    number_of_time_step = 10**np.arange(3)
    time_step_sizes     = 1 / number_of_time_step
    error = np.zeros(time_step_sizes.size)

    for i in range(time_step_sizes.size):
        test_obj = test()
        for j in range(number_of_time_step[i]):
            swss(test_obj, op1, op2, time_step_sizes[i])
        error[i] = abs(af.sum(test_obj.f) - np.exp(1) + 1)

    poly = np.polyfit(np.log10(number_of_time_step), np.log10(error), 1)
    assert (abs(poly[0] + 2) < 0.2)

def test_jia_split_operations():

    number_of_time_step = 10**np.arange(3)
    time_step_sizes     = 1 / number_of_time_step
    error = np.zeros(time_step_sizes.size)

    for i in range(time_step_sizes.size):
        test_obj = test()
        for j in range(number_of_time_step[i]):
            jia(test_obj, op1, op2, time_step_sizes[i])
        error[i] = abs(af.sum(test_obj.f) - np.exp(1) + 1)

    poly = np.polyfit(np.log10(number_of_time_step), np.log10(error), 1)
    assert (abs(poly[0] + 4) < 0.2)
