#!/usr/bin/env python 
# -*- coding: utf-8 -*-

# This test ensures that the RK4 implementation is 4th order in time
# For this, we consider the test problem df/dt = f 
# We integrate till t = 1 and compare the results with the expected analytic solution f = e^t

from lib.nonlinear_solver.solve_source_sink import RK4
import numpy as np
import arrayfire as af 

class test(object):
  def __init__(self):
    self.f = af.to_array(np.array([1.0]))
  
  def g(self):
    return(self.f)

def test_RK4():
  number_of_time_step = 10**np.arange(4)
  time_step_sizes     = 1/number_of_time_step
  error               = np.zeros(time_step_sizes.size)

  for i in range(time_step_sizes.size):
    test_obj = test()
    for j in range(number_of_time_step[i]):
      RK4(test_obj, time_step_sizes[i])
    error[i] = abs(af.sum(test_obj.f) - np.exp(1))

  print(error)
  poly = np.polyfit(np.log10(number_of_time_step), np.log10(error), 1)
  print(poly)
  assert(abs(poly[0]+4)<0.2)