#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import numpy as np
from scipy.fftpack import fftfreq
import arrayfire as af

from lib.linear_solver.linear_solver import linear_solver as linear_solver

calculate_k_center = linear_solver._calculate_k

class test():
  def __init__(self):
    self.q1_start = np.random.randint(0, 5)
    self.q2_start = np.random.randint(0, 5)

    self.q1_end = np.random.randint(5, 10)
    self.q2_end = np.random.randint(5, 10)

    self.N_q1 = np.random.randint(32, 128)
    self.N_q2 = np.random.randint(32, 128)

    self.dq1 = (self.q1_end - self.q1_start)/self.N_q1
    self.dq2 = (self.q2_end - self.q2_start)/self.N_q2

    self.N_p1 = np.random.randint(5, 20)
    self.N_p2 = np.random.randint(5, 20)
    self.N_p3 = np.random.randint(5, 20)

def test_calculate_k():
  test_obj = test()

  k1, k2 = calculate_k_center(test_obj)

  k1_expected = af.to_array(2 * np.pi * fftfreq(test_obj.N_q1, test_obj.dq1))
  k2_expected = af.to_array(2 * np.pi * fftfreq(test_obj.N_q2, test_obj.dq2))

  k1_expected = af.tile(k1_expected, 1, test_obj.N_q2, test_obj.N_p1 * test_obj.N_p2 * test_obj.N_p3)
  k2_expected = af.tile(af.reorder(k2_expected), test_obj.N_q1, 1, test_obj.N_p1 * test_obj.N_p2 * test_obj.N_p3)

  assert(af.sum(af.abs(k1_expected - k1)) + af.sum(af.abs(k2_expected - k2))==0)