#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This test ensures that the implementation of time-steppers is accurate
For this, we consider the test problem df/dt = f + 1
We integrate till t = 1 and compare the results with the expected
analytic solution f = e^t - 1
"""

import arrayfire as af
import numpy as np

from bolt.lib.nonlinear_solver.timestepper_source \
    import RK4_step
from bolt.lib.nonlinear_solver.timestepper \
    import _strang_split_operations,_lie_split_operations, _swss_split_operations, _jia_split_operations

class test(object):
    def __init__(self):
        self.f = af.to_array(np.array([0]))
        
        self.E1 = af.to_array(np.array([0]))
        self.E2 = af.to_array(np.array([0]))
        self.E3 = af.to_array(np.array([0]))

        self.B1 = af.to_array(np.array([0]))
        self.B2 = af.to_array(np.array([0]))
        self.B3 = af.to_array(np.array([0]))

        self.testing_source_flag   = True 
        self.performance_test_flag = False

    def _term1(self, f):
        return (f)

    def _term2(self, f):
        return (1)

    def _apply_BC_distribution_function(self):
        return None

def op1(self, dt):
    self._source = self._term1
    return(RK4_step(self, dt))

def op2(self, dt):
    self._source = self._term2
    return(RK4_step(self, dt))

def test_strang_split_operations():

    number_of_time_step = 10**np.arange(3)
    time_step_sizes     = 1 / number_of_time_step
    error = np.zeros(time_step_sizes.size)

    for i in range(time_step_sizes.size):
        test_obj = test()
        for j in range(number_of_time_step[i]):
            _strang_split_operations(test_obj, op1, op2, time_step_sizes[i])
        error[i] = abs(af.sum(test_obj.f) - np.exp(1) + 1)

    poly = np.polyfit(np.log10(number_of_time_step), np.log10(error), 1)
    assert (abs(poly[0] + 2) < 0.2)

def test_lie_split_operations():

    number_of_time_step = 10**np.arange(3)
    time_step_sizes     = 1 / number_of_time_step
    error = np.zeros(time_step_sizes.size)

    for i in range(time_step_sizes.size):
        test_obj = test()
        for j in range(number_of_time_step[i]):
            _lie_split_operations(test_obj, op1, op2, time_step_sizes[i])
        error[i] = abs(af.sum(test_obj.f) - np.exp(1) + 1)

    poly = np.polyfit(np.log10(number_of_time_step), np.log10(error), 1)
    assert (abs(poly[0] + 1) < 0.2)

def test_swss_split_operations():

    number_of_time_step = 10**np.arange(3)
    time_step_sizes     = 1 / number_of_time_step
    error = np.zeros(time_step_sizes.size)

    for i in range(time_step_sizes.size):
        test_obj = test()
        for j in range(number_of_time_step[i]):
            _swss_split_operations(test_obj, op1, op2, time_step_sizes[i])
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
            _jia_split_operations(test_obj, op1, op2, time_step_sizes[i])
        error[i] = abs(af.sum(test_obj.f) - np.exp(1) + 1)

    poly = np.polyfit(np.log10(number_of_time_step), np.log10(error), 1)
    assert (abs(poly[0] + 4) < 0.2)
